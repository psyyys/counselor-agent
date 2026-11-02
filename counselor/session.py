"""会谈流水线（框架采用 PsychEval，见 docs/咨询师A设计.md）。

会谈开始   start()  ：第 1 次用固定议程；之后调用 1 次模型生成议程 → 检索技能和手册（本次会谈内固定）
每轮对话   turn()   ：安全预检 → 检索范例 → 主模型（思考+回复）→ 督导审查 → 不合格就重写一次
会谈结束   end()    ：总结 + 更新档案（第 1 次还要生成疗程计划）→ 判断是否进入下一阶段
"""
import re
from datetime import datetime

from . import prompts as P
from .config import INDEX_DIR, USERS_DIR
from .llm import parse_json
from .memory import STAGES, UserStore
from .retrieval import BM25Index
from .safety import HOTLINE_TEXT, precheck

START_MARK = "（会谈开始，请咨询师先开口）"


class Indexes:
    def __init__(self, root=INDEX_DIR):
        self.exemplars = BM25Index.load(root / "exemplars.pkl")
        self.skills = BM25Index.load(root / "skills.pkl")
        mp = root / "manuals.pkl"
        self.manuals = BM25Index.load(mp) if mp.exists() else None


THINK_FIELDS = ("assessment", "client_state", "skill", "strategy", "risk")


def parse_reply(raw: str) -> dict:
    """回复用 <reply> 包住。不用 <think>：DeepSeek 把 </think> 当成自己的特殊标记，写到那里就停了。
    兼容旧格式：有 </think> 就按它切；都没有就把思考字段整块删掉，剩下的当回复。"""
    out = {}
    for f in THINK_FIELDS:
        m = re.search(rf"<{f}>(.*?)</{f}>", raw, re.S)
        out[f] = m.group(1).strip() if m else ""
    if "<reply>" in raw:
        reply = raw.split("<reply>", 1)[1].split("</reply>", 1)[0]
    elif "</think>" in raw:
        reply = raw.split("</think>", 1)[1]
    else:
        reply = re.sub(rf"<({'|'.join(THINK_FIELDS)})>.*?</\1>", "", raw, flags=re.S)
        reply = re.sub(r"</?think>", "", reply)
    out["reply"] = reply.strip()
    return out


def count_questions(text: str) -> int:
    return len(re.findall(r"[？?]", text))


class Counselor:
    def __init__(self, llm, cfg: dict, indexes: Indexes, exclude_cases: set[str] | None = None, users_root=USERS_DIR):
        self.llm, self.cfg, self.ix, self.users_root = llm, cfg, indexes, users_root
        # 评测时用 PsychEval 案例当模拟来访者，必须把这个案例从范例库里排除，否则等于偷看答案
        self.exclude_cases = exclude_cases or set()

    # ---------- 会谈开始 ----------
    def start(self, user: str) -> dict:
        store = UserStore(user, self.users_root)
        state = store.state()
        if state["active_session"]:
            sess = store.session(state["active_session"])
            return {"session": sess, "resumed": True}

        n = state["sessions_done"] + 1
        stage = STAGES[state["stage_index"]]
        if n == 1:
            agenda = dict(P.FIRST_SESSION_AGENDA)
        else:
            agenda = self._make_agenda(store, n, stage)
        agenda["skills"] = self._pick_skills(agenda, stage)
        agenda["manual_chunks"] = self._pick_manuals(agenda)

        sess = store.new_session(n, stage, agenda)
        state["active_session"] = n
        store.save_state(state)
        first = self._respond(store, sess, START_MARK, record_user=False)
        return {"session": store.session(n), "resumed": False, "first": first}

    def _make_agenda(self, store: UserStore, n: int, stage: str) -> dict:
        last = store.last_finished_session()
        user_msg = (f"【个案档案】\n{P.fmt_json(store.profile())}\n\n【疗程计划】\n{P.fmt_json(store.plan())}\n\n"
                    f"【以往会谈】\n{P.fmt_history(store.all_summaries())}\n\n"
                    f"【上次会谈的遗留问题】\n{P.fmt_json(last['summary'].get('client_state_analysis') if last and last.get('summary') else None)}\n"
                    f"上次使用的疗法：{last['agenda'].get('theory_select') if last else '无'}")
        raw = self.llm.chat(self.cfg["main_model"], [
            {"role": "system", "content": P.AGENDA_SYSTEM.format(session_number=n, stage=stage)},
            {"role": "user", "content": user_msg}], temperature=0.3, json_mode=True)
        agenda = parse_json(raw)
        if agenda.get("theory_select") not in P.THEORY_NAMES:
            agenda["theory_select"] = "CBT"
        return agenda

    def _pick_skills(self, agenda: dict, stage: str) -> list[dict]:
        """PsychEval E.5 用模型从 4500 个技能里挑；本项目改编为按议程检索（省一次大调用）。
        CBT 技能来自 PsychEval；DBT / ACT 技能由 build_skills.py 从手册抽取。只取本次所选疗法的技能。"""
        theory = agenda.get("theory_select", "CBT")
        query = agenda.get("stage_title", "") + " " + " ".join(agenda.get("objective", []))
        hits = self.ix.skills.search(query, k=self.cfg["skills_per_session"],
                                     where=lambda s: s.get("theory", "CBT") == theory and stage in s["stages"])
        keep = ("skill_id", "name", "description", "when_to_use", "trigger", "meta")
        return [{k: s.get(k, "") for k in keep} for _, s in hits]

    def _pick_manuals(self, agenda: dict) -> list[dict]:
        if not self.ix.manuals:
            return []
        theory = agenda.get("theory_select", "CBT").lower()
        per_q = max(1, self.cfg["manual_chunks_per_session"] // max(len(agenda.get("manual_queries_en", [])), 1))
        seen, out = set(), []
        for q in agenda.get("manual_queries_en", []):
            for _, c in self.ix.manuals.search(q, k=per_q + 2, where=lambda d: d["theory"] == theory):
                key = (c["source"], c["page"])
                if key not in seen:
                    seen.add(key)
                    out.append(c)
                    break
        return out[: self.cfg["manual_chunks_per_session"]]

    # ---------- 每轮 ----------
    def turn(self, user: str, text: str, note: str = "", voice_cues: dict | None = None) -> dict:
        """note：附加给咨询师的系统提示（评测时用来提醒"会谈时间快到了"）。
        voice_cues：语音通话时浏览器算出的声音线索 {"summary": "...", "metrics": {...}}。"""
        store = UserStore(user, self.users_root)
        n = store.state()["active_session"]
        if not n:
            raise RuntimeError("没有进行中的会谈，请先开始会谈")
        return self._respond(store, store.session(n), text.strip(), record_user=True, note=note, voice_cues=voice_cues)

    def _respond(self, store: UserStore, sess: dict, text: str, record_user: bool, note: str = "",
                 voice_cues: dict | None = None) -> dict:
        agenda, stage = sess["agenda"], sess["stage"]
        crisis_hits = precheck(text) if record_user else []
        cue_text = (voice_cues or {}).get("summary", "").strip()[:300]
        if record_user:
            entry = {"role": "client", "text": text, "time": _now()}
            if cue_text:
                entry["voice_cues"] = {"summary": cue_text, "metrics": (voice_cues or {}).get("metrics", {})}
            sess["transcript"].append(entry)

        exemplars = self._pick_exemplars(text, stage, opening=not record_user)
        theory = agenda.get("theory_select", "CBT")
        system = P.COUNSELOR_SYSTEM.format(
            session_number=sess["session_number"], stage=stage, theory_name=P.THEORY_NAMES[theory],
            stage_title=agenda.get("stage_title", ""), objectives=P.fmt_objectives(agenda),
            crisis_note=(P.CRISIS_NOTE if crisis_hits else "") + (f"\n【提示】{note}" if note else "")
                        + (P.VOICE_CUE_NOTE.format(cues=cue_text) if cue_text else ""))
        context = P.CONTEXT_TEMPLATE.format(
            profile=P.fmt_json(store.profile()), history=P.fmt_history(store.all_summaries()),
            skills=P.fmt_skills(agenda.get("skills", [])), manuals=P.fmt_manuals(agenda.get("manual_chunks", [])),
            exemplars=P.fmt_exemplars(exemplars))
        messages = [{"role": "system", "content": system}, {"role": "system", "content": context}]
        messages += self._history_messages(sess)
        if not record_user:
            messages.append({"role": "user", "content": text})

        attempts = []
        draft = parse_reply(self.llm.chat(self.cfg["main_model"], messages, self.cfg["main_temperature"]))
        for i in range(self.cfg["max_revisions"] + 1):
            verdict = self._critique(sess, draft, cue_text)
            attempts.append({"draft": draft, "verdict": verdict})
            if verdict["pass"] or i == self.cfg["max_revisions"]:
                break
            revise = P.REVISE_NOTE.format(draft=draft["reply"], failed="；".join(verdict["failed"]),
                                          advice=verdict.get("advice", ""))
            draft = parse_reply(self.llm.chat(self.cfg["main_model"],
                                              messages + [{"role": "user", "content": revise}],
                                              self.cfg["main_temperature"]))

        final = attempts[-1]["draft"]
        reply = final["reply"]
        # 热线单独存放、不进模型看到的历史（否则模型会照抄，越积越多）。
        # 出现危机词必定显示；只有模型判断高风险时，最近 6 句咨询师发言里显示过就不重复
        recent = [t for t in sess["transcript"] if t["role"] == "counselor"][-6:]
        show_hotline = bool(crisis_hits) or (final["risk"].startswith("高") and not any(t.get("hotline") for t in recent))
        sess["transcript"].append({"role": "counselor", "text": reply, "hotline": show_hotline,
                                   "think": {k: final[k] for k in THINK_FIELDS}, "time": _now()})
        sess.setdefault("flags", [])
        if crisis_hits or final["risk"].startswith("高"):
            sess["flags"].append({"turn": len(sess["transcript"]) - 1, "crisis_words": crisis_hits, "risk": final["risk"]})
        store.save_session(sess)
        return {"reply": reply, "hotline": HOTLINE_TEXT if show_hotline else "", "inspector": {
            "think": {k: final[k] for k in ("assessment", "client_state", "skill", "strategy", "risk")},
            "crisis_words": crisis_hits, "hotline_shown": show_hotline,
            "voice_cues": (voice_cues if cue_text else None),
            "attempts": [{"reply": a["draft"]["reply"], "verdict": a["verdict"]} for a in attempts],
            "exemplars": [{"case": e["case"], "session": e["session"], "client": e["client"][:120],
                           "reply": e["reply"][:160]} for e in exemplars],
            "usage": dict(self.llm.usage)}}

    def _pick_exemplars(self, text: str, stage: str, opening: bool) -> list[dict]:
        k, ex = self.cfg["exemplars_per_turn"], self.exclude_cases
        if opening:
            hits = self.ix.exemplars.search("你好 欢迎 开始 会谈 回顾 上周", k=k, where=lambda d: d["position"] == "开场"
                                            and d["stage"] == stage and d["case"] not in ex)
        else:
            hits = self.ix.exemplars.search(text, k=k, where=lambda d: d["stage"] == stage and d["case"] not in ex)
        return [d for _, d in hits]

    def _history_messages(self, sess: dict) -> list[dict]:
        msgs = []
        for t in sess["transcript"][-self.cfg["history_messages"]:]:
            role = "user" if t["role"] == "client" else "assistant"
            msgs.append({"role": role, "content": t["text"].replace(HOTLINE_TEXT, "").strip()})   # 兼容旧记录
        return msgs

    def _critique(self, sess: dict, draft: dict, cue_text: str = "") -> dict:
        # 先用代码查"最多一个问题"，省得为这种问题再请督导
        if count_questions(draft["reply"]) > 1:
            return {"pass": False, "failed": ["A1：问了不止一个问题"], "advice": "只保留最关键的一个问题。", "by": "code"}
        if not draft["reply"]:
            return {"pass": False, "failed": ["格式：没有回复内容"], "advice": "按格式先写思考字段，再用 <reply></reply> 写回复。", "by": "code"}
        agenda = sess["agenda"]
        theory = agenda.get("theory_select", "CBT")
        recent = "\n".join(f"{'来访者' if t['role'] == 'client' else '咨询师'}：{t['text']}"
                           for t in sess["transcript"][-6:])
        user_msg = f"【最近的对话】\n{recent or '（会谈刚开始）'}\n\n【待审查的咨询师回复】\n{draft['reply']}\n\n【咨询师的思考】\n{draft['strategy']}"
        if cue_text:
            user_msg += f"\n\n【咨询师拿到的声音线索】\n{cue_text}"
        try:
            raw = self.llm.chat(self.cfg["critic_model"], [
                {"role": "system", "content": P.CRITIC_SYSTEM.format(
                    theory_name=P.THEORY_NAMES[theory], stage=sess["stage"], stage_title=agenda.get("stage_title", ""))},
                {"role": "user", "content": user_msg}], self.cfg["critic_temperature"], json_mode=True)
            v = parse_json(raw)
            return {"pass": bool(v.get("pass")), "failed": v.get("failed") or [], "advice": v.get("advice", ""), "by": "critic"}
        except Exception as e:      # 督导失败不能挡住对话
            return {"pass": True, "failed": [], "advice": f"督导调用失败：{e}", "by": "error"}

    # ---------- 会谈结束 ----------
    def end(self, user: str) -> dict:
        store = UserStore(user, self.users_root)
        state = store.state()
        n = state["active_session"]
        if not n:
            raise RuntimeError("没有进行中的会谈")
        sess = store.session(n)
        agenda, theory = sess["agenda"], sess["agenda"].get("theory_select", "CBT")
        transcript = "\n".join(f"{'来访者' if t['role'] == 'client' else '咨询师'}：{t['text']}" for t in sess["transcript"])

        summary = parse_json(self.llm.chat(self.cfg["main_model"], [
            {"role": "system", "content": P.SUMMARY_SYSTEM.format(
                session_number=n, stage=sess["stage"], theory_name=P.THEORY_NAMES[theory],
                stage_title=agenda.get("stage_title", ""), objectives=P.fmt_objectives(agenda))},
            {"role": "user", "content": f"【个案档案】\n{P.fmt_json(store.profile())}\n\n【以往会谈】\n"
                                        f"{P.fmt_history(store.all_summaries())}\n\n【本次逐字稿】\n{transcript}"}],
            temperature=0.2, json_mode=True))

        new_profile = parse_json(self.llm.chat(self.cfg["main_model"], [
            {"role": "system", "content": P.memory_system()},
            {"role": "user", "content": f"【旧档案】\n{P.fmt_json(store.profile())}\n\n【本次逐字稿】\n{transcript}"}],
            temperature=0.0, json_mode=True))
        store.save_profile(new_profile)

        if n == 1:
            plan = parse_json(self.llm.chat(self.cfg["main_model"], [
                {"role": "system", "content": P.PLAN_SYSTEM},
                {"role": "user", "content": f"【个案档案】\n{P.fmt_json(new_profile)}\n\n【第 1 次会谈总结】\n{P.fmt_json(summary)}"}],
                temperature=0.3, json_mode=True))
            store.save_plan(plan)

        sess["summary"], sess["status"], sess["ended_at"] = summary, "done", _now()
        store.save_session(sess)
        advanced = bool(summary.get("progress", {}).get("advance_stage")) and state["stage_index"] < len(STAGES) - 1
        if advanced:
            state["stage_index"] += 1
        state["sessions_done"], state["active_session"] = n, None
        store.save_state(state)
        return {"summary": summary, "stage_advanced": advanced, "next_stage": STAGES[state["stage_index"]],
                "usage": dict(self.llm.usage)}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
