"""离线测试：用假模型跑通 开始 → 两轮对话（含一次督导退回、一次危机词）→ 结束。"""
import json

import pytest

from counselor import session as S
from counselor.build_index import parse_counselor_turn
from counselor.config import DEFAULTS
from counselor.llm import FakeLLM, parse_json
from counselor.memory import STAGES, UserStore, empty_profile
from counselor.retrieval import BM25Index, tokenize
from counselor.safety import precheck


def make_indexes():
    ex = [{"case": "1", "session": 1, "turn": 0, "stage": STAGES[0], "position": "开场", "client": "（会谈开始）",
           "assessment": "初次见面", "client_state": "紧张", "skill": "32:共情", "strategy": "欢迎", "reply": "你好，欢迎你来。"},
          {"case": "1", "session": 1, "turn": 4, "stage": STAGES[0], "position": "中段", "client": "我最近总是睡不着，很焦虑",
           "assessment": "睡眠问题", "client_state": "焦虑", "skill": "68202:开放式提问", "strategy": "了解情境", "reply": "能说说最近一次睡不着时的情况吗？"}]
    sk = [{"skill_id": "68202", "name": "开放式提问", "description": "邀请来访者讲述", "when_to_use": "收集资料",
           "trigger": "初次会谈", "meta": "会谈技术", "stages": [STAGES[0]]}]
    ix = S.Indexes.__new__(S.Indexes)
    ix.exemplars = BM25Index(ex, [e["client"] for e in ex])
    ix.skills = BM25Index(sk, [s["name"] + s["description"] for s in sk])
    ix.manuals = None
    return ix


def fake_handler():
    state = {"main": 0}

    def handler(model, messages, json_mode):
        sys = messages[0]["content"]
        if "临床督导。请审查" in sys:
            reply = messages[-1]["content"].split("【待审查的咨询师回复】")[1]
            ok = "第二版" in reply or "欢迎" in reply or "安全" in reply
            return json.dumps({"pass": ok, "failed": [] if ok else ["B6：过早给建议"], "advice": "先多了解情况"}, ensure_ascii=False)
        if "会谈写总结" in sys:
            return json.dumps({"session_summary_abstract": "来访者谈到失眠。", "goal_assessment": {},
                               "client_state_analysis": {}, "homework": ["情绪记录"],
                               "progress": {"advance_stage": True, "reason": "初始评估完成"}}, ensure_ascii=False)
        if "档案管理员" in sys:
            p = empty_profile()
            p["main_problem"] = "失眠、焦虑"
            return json.dumps(p, ensure_ascii=False)
        if "疗程规划者" in sys:
            return json.dumps({"total_sessions": 6, "stages": [], "sessions": []}, ensure_ascii=False)
        state["main"] += 1
        last = messages[-1]["content"]
        if "没有通过督导审查" in last:
            body = "第二版：听起来这段时间睡不好让你很难受。最近一次是什么情况？"
        elif "不想活" in last:
            body = "谢谢你愿意告诉我，我很关心你的安全。你现在有伤害自己的打算吗？"
        elif "会谈开始" in last:
            body = "你好，欢迎你来。我可以怎么称呼你？"
        else:
            body = "你应该早点睡觉。"
        risk = "高" if "不想活" in last else "无"
        return f"<think><assessment>a</assessment><client_state>c</client_state><skill>68202:开放式提问</skill><strategy>s</strategy><risk>{risk}</risk></think>{body}"
    return handler


@pytest.fixture
def counselor(tmp_path):
    llm = FakeLLM(fake_handler())
    return S.Counselor(llm, dict(DEFAULTS), make_indexes(), users_root=tmp_path), llm, tmp_path


def test_full_session(counselor):
    c, llm, root = counselor
    r = c.start("测试者")
    assert r["first"]["reply"].startswith("你好")
    assert r["session"]["agenda"]["theory_select"] == "CBT"
    assert r["session"]["agenda"]["skills"][0]["skill_id"] == "68202"

    # 第一版是说教，督导退回 → 第二版通过
    t1 = c.turn("测试者", "我最近总是睡不着，很焦虑")
    assert len(t1["inspector"]["attempts"]) == 2
    assert t1["reply"].startswith("第二版")
    assert t1["inspector"]["exemplars"]

    # 危机词：附热线
    t2 = c.turn("测试者", "有时候觉得不想活了")
    assert t2["inspector"]["crisis_words"]
    assert "12356" in t2["hotline"] and "12356" not in t2["reply"]
    # 热线不进模型看到的历史
    assert all("12356" not in m["content"] for m in c._history_messages(UserStore("测试者", root).session(1)))

    e = c.end("测试者")
    assert e["stage_advanced"] and e["next_stage"] == STAGES[1]
    store = UserStore("测试者", root)
    assert store.profile()["main_problem"] == "失眠、焦虑"
    assert store.plan()["total_sessions"] == 6
    assert store.state() == {"sessions_done": 1, "stage_index": 1, "active_session": None}
    sess = store.session(1)
    assert sess["status"] == "done" and sess["flags"]


def test_voice_cues_reach_prompt_and_critic(counselor):
    c, llm, root = counselor
    c.start("声音测试")
    cues = {"summary": "这句比你本次平均慢了约 35%，中间有 2 次超过 1 秒的停顿", "metrics": {"rate": 2.1}}
    t = c.turn("声音测试", "我最近总是睡不着，很焦虑", voice_cues=cues)
    main_calls = [x for x in llm.calls if "认知行为流派的心理咨询师" in x["messages"][0]["content"]]
    assert "停顿" in main_calls[-1]["messages"][0]["content"] and "不要据此给来访者下情绪结论" in main_calls[-1]["messages"][0]["content"]
    critic_calls = [x for x in llm.calls if "临床督导。请审查" in x["messages"][0]["content"]]
    assert "声音线索" in critic_calls[-1]["messages"][-1]["content"]
    assert t["inspector"]["voice_cues"]["summary"].startswith("这句比你")
    client_entry = [x for x in UserStore("声音测试", root).session(1)["transcript"] if x["role"] == "client"][-1]
    assert client_entry["voice_cues"]["metrics"] == {"rate": 2.1}
    # 打字（没有声音线索）时提示词里不出现
    c.turn("声音测试", "嗯")
    main_calls = [x for x in llm.calls if "认知行为流派的心理咨询师" in x["messages"][0]["content"]]
    assert "声音线索" not in main_calls[-1]["messages"][0]["content"]


def test_code_check_rejects_two_questions(counselor):
    c, _, _ = counselor
    v = c._critique({"agenda": {}, "stage": STAGES[0], "transcript": []}, {"reply": "你好吗？你睡得怎样？", "strategy": ""})
    assert not v["pass"] and v["by"] == "code"


def test_parse_reply_without_think_tags():
    # DeepSeek 接口会吞掉 <think></think>，只留内部字段
    raw = "<assessment>初次见面</assessment><client_state>紧张</client_state><skill>32:共情</skill><strategy>欢迎</strategy><risk>无</risk>你好，欢迎你来。"
    r = S.parse_reply(raw)
    assert r["reply"] == "你好，欢迎你来。" and r["assessment"] == "初次见面" and r["risk"] == "无"
    assert S.parse_reply("<think><risk>低</risk></think>好的")["reply"] == "好的"
    assert S.parse_reply("<risk>低</risk>\n<reply>嗯，我在听。</reply>")["reply"] == "嗯，我在听。"
    assert S.parse_reply("<risk>低</risk>")["reply"] == ""


def test_records_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from counselor import server
    monkeypatch.setattr(UserStore.__init__, "__defaults__", (tmp_path,))
    store = UserStore("记录测试", tmp_path)
    store.save_state({"sessions_done": 1, "stage_index": 0, "active_session": None})
    store.save_session({"session_number": 1, "stage": STAGES[0], "agenda": {"theory_select": "CBT", "stage_title": "初始评估"},
                        "transcript": [{"role": "client", "text": "你好"}], "summary": {"homework": ["情绪记录"]}, "status": "done"})
    c = TestClient(server.app)
    assert c.post("/api/records/mood", json={"user": "记录测试", "session_number": 1, "phase": "before", "value": 4}).status_code == 200
    assert c.post("/api/records/mood", json={"user": "记录测试", "session_number": 1, "phase": "after", "value": 11}).status_code == 400
    c.post("/api/records/homework", json={"user": "记录测试", "session_number": 1, "index": 0, "done": True})
    item = c.post("/api/records/journal", json={"user": "记录测试", "text": "今天睡得好一点"}).json()
    r = c.get("/api/records/记录测试").json()
    s = r["sessions"][0]
    assert s["mood"] == {"before": 4} and s["homework"] == [{"text": "情绪记录", "done": True}] and s["turns"] == 1
    assert r["journal"][0]["text"] == "今天睡得好一点"
    c.post("/api/records/journal/delete", json={"user": "记录测试", "id": item["id"]})
    assert c.get("/api/records/记录测试").json()["journal"] == []


def test_clean_for_speech():
    from counselor.voice import clean_for_speech
    assert clean_for_speech("**重点**\n- 第一条\n\n\n- 第二条") == "重点\n第一条\n第二条"


def test_eval_formatting():
    from experiments.judge import format_dialogue, format_profile
    d = format_dialogue([{"role": "Counselor", "text": "<think>x</think>你好"}, {"role": "client", "text": "嗯"}])
    assert d == "counselor: 你好\nclient: 嗯"
    p = format_profile({"core_beliefs": ["我不行"], "static_traits": {"name": "A"}})
    assert p["core_brief"] == ["我不行"] and p["static_traits"]["name"] == "A"


def test_helpers():
    assert "睡不" in tokenize("我睡不着") and "anxiety" in tokenize("Anxiety and sleep")
    assert precheck("我真的不想活了") and not precheck("今天天气不错")
    t = parse_counselor_turn("<think><assessment>评估</assessment><strategy>策略</strategy></think>你好")
    assert t["assessment"] == "评估" and t["reply"] == "你好"
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
