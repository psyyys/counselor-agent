"""从 DBT / ACT 手册抽取技能库（PsychEval 的做法：用强模型把教材解析成 元技能 → 微技能，
每个微技能带"何时使用"和"触发条件"）。结果存 data/skills/{dbt,act}.json，可以人工查看、修改。

    python -m counselor.build_skills            # 抽取 DBT 和 ACT
    python -m counselor.build_skills act        # 只抽 ACT

抽取完要重跑 python -m counselor.build_index，技能才会进检索库。
"""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .build_index import _book_title, _is_toc_or_index
from .config import DATA, KB_RAW, load_config
from .llm import LLM, parse_json
from .memory import STAGES

SKILLS_DIR = DATA / "skills"
WINDOW_CHARS = 9000

EXTRACT_SYSTEM = """你是{theory_full}的资深培训师，正在把治疗手册整理成"技能库"，供 AI 咨询师在一对一文字会谈中使用（PsychEval 技能库格式）。

请从下面这段手册原文中，抽取**咨询师在会谈里可以直接做的技能**：
- 只抽原文里明确讲到的技能或技术，不要自己发明，不要补充原文没有的内容
- 只抽适合一对一会谈的；纯团体管理、电话教练、治疗师顾问团队等内容不要
- 原文只是理论、研究或例子、没有可操作技能时，返回空列表
- 用中文写，专有名词后面括号保留英文原名，例如"TIPP 技能（TIPP）"

每个技能的字段：
- meta：元技能（所属模块/类别），例如"痛苦耐受·危机生存技能"
- name：技能名称（15 字以内，动宾结构，例如"引导来访者用冷水降低生理唤醒"）
- description：咨询师具体怎么做、怎么说（60–120 字）
- when_to_use：在会谈的什么时机使用
- trigger：来访者出现什么信号时使用
- stages：适用阶段，从这三个里选一个或多个：{stages}
- source_term：原文中的英文名称

只输出 JSON：{{"skills": [ {{...}}, ... ]}}"""

THEORY_FULL = {"dbt": "辩证行为疗法（DBT）", "act": "接纳承诺疗法（ACT）"}


def windows(theory: str):
    from pypdf import PdfReader
    for pdf in sorted((KB_RAW / theory).glob("*.pdf")):
        title = _book_title(pdf.stem)
        reader = PdfReader(str(pdf))
        n = len(reader.pages)
        if n < 20:          # Extra Bits 这类链接合集跳过
            continue
        front, back = int(n * 0.06), n - int(n * 0.04)
        buf, start = "", None
        for pno in range(front + 1, back + 1):
            try:
                text = reader.pages[pno - 1].extract_text() or ""
            except Exception:
                continue
            text = re.sub(r"[ \t]+", " ", text).strip()
            if len(text) < 200 or _is_toc_or_index(text):
                continue
            if start is None:
                start = pno
            buf += f"\n[第{pno}页]\n{text}"
            if len(buf) >= WINDOW_CHARS:
                yield {"source": title, "pages": f"{start}-{pno}", "text": buf}
                buf, start = "", None
        if buf:
            yield {"source": title, "pages": f"{start}-{back}", "text": buf}


def extract(llm, model, theory, win) -> list[dict]:
    system = EXTRACT_SYSTEM.format(theory_full=THEORY_FULL[theory], stages="、".join(STAGES))
    raw = llm.chat(model, [{"role": "system", "content": system},
                           {"role": "user", "content": f"【手册】{win['source']}（第 {win['pages']} 页）\n{win['text']}"}],
                   temperature=0.0, json_mode=True)
    out = []
    for s in parse_json(raw).get("skills", []):
        if not s.get("name") or not s.get("description"):
            continue
        s["stages"] = [x for x in s.get("stages", []) if x in STAGES] or [STAGES[1]]
        s.update({"theory": theory.upper(), "source": win["source"], "pages": win["pages"]})
        out.append(s)
    return out


def dedupe(skills: list[dict]) -> list[dict]:
    """同一元技能下名称相同、或英文原名相同的视为重复；保留描述更长的，阶段取并集。"""
    seen: dict[tuple, dict] = {}
    for s in skills:
        key = (s.get("meta", ""), re.sub(r"\W", "", s["name"]))
        term = re.sub(r"\W", "", (s.get("source_term") or "").lower())
        key2 = (s.get("meta", ""), term) if term else None
        hit = seen.get(key) or (seen.get(key2) if key2 else None)
        if hit:
            hit["stages"] = sorted(set(hit["stages"]) | set(s["stages"]), key=STAGES.index)
            if len(s["description"]) > len(hit["description"]):
                hit.update({k: s[k] for k in ("description", "when_to_use", "trigger")})
            continue
        seen[key] = s
        if key2:
            seen[key2] = s
    uniq = list({id(v): v for v in seen.values()}.values())
    for i, s in enumerate(uniq, 1):
        s["skill_id"] = f"{s['theory']}{i:04d}"
    return uniq


def main():
    theories = [a for a in sys.argv[1:] if a in THEORY_FULL] or ["dbt", "act"]
    cfg = load_config()
    llm = LLM(cfg)
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    for theory in theories:
        # 每个窗口抽完立刻追加到 _raw.jsonl；中途停了再运行会跳过已完成的窗口
        raw_path = SKILLS_DIR / f"{theory}_raw.jsonl"
        done, raw = set(), []
        if raw_path.exists():
            for line in raw_path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                done.add((rec["source"], rec["pages"]))
                raw.extend(rec["skills"])
        wins = [w for w in windows(theory) if (w["source"], w["pages"]) not in done]
        print(f"{theory.upper()}：已完成 {len(done)} 个窗口，还剩 {len(wins)} 个，开始抽取……", flush=True)
        failed = 0
        with ThreadPoolExecutor(max_workers=6) as pool, open(raw_path, "a", encoding="utf-8") as fout:
            futs = {pool.submit(extract, llm, cfg["main_model"], theory, w): w for w in wins}
            for i, f in enumerate(as_completed(futs), 1):
                w = futs[f]
                try:
                    got = f.result()
                except Exception as e:
                    failed += 1
                    print(f"  窗口失败 {w['pages']}: {e}", flush=True)
                    continue
                raw.extend(got)
                fout.write(json.dumps({"source": w["source"], "pages": w["pages"], "skills": got}, ensure_ascii=False) + "\n")
                fout.flush()
                if i % 20 == 0:
                    print(f"  {i}/{len(wins)}，累计 {len(raw)} 条", flush=True)
        skills = dedupe(raw)
        (SKILLS_DIR / f"{theory}.json").write_text(json.dumps(skills, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{theory.upper()} 完成：原始 {len(raw)} 条 → 去重后 {len(skills)} 条，失败窗口 {failed} 个", flush=True)
    print("用量：", llm.usage, flush=True)


if __name__ == "__main__":
    main()
