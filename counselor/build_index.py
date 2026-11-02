"""建三个检索库，结果存到 data/index/。只需运行一次（换数据后重跑）：

    python -m counselor.build_index

1. 范例库 exemplars.pkl：PsychEval CBT 对话里每一句咨询师发言（连同前一句来访者的话和 <think>）
2. 技能库 skills.pkl：PsychEval CBT 会谈里出现过的全部微技能，记录各自出现在哪些阶段
3. 手册库 manuals.pkl：data/kb_raw/{cbt,dbt,act}/*.pdf 按页抽文字后切块
"""
import json
import re
import sys
from pathlib import Path

from .config import DATA, INDEX_DIR, KB_RAW, PSYCHEVAL_CBT

SKILLS_DIR = DATA / "skills"
from .retrieval import BM25Index

THINK_FIELDS = ("assessment", "client_state", "skill", "strategy")


def parse_counselor_turn(text: str) -> dict:
    """把 '<think><assessment>..</assessment>..</think>回复' 拆成字段。"""
    out = {}
    for f in THINK_FIELDS:
        m = re.search(rf"<{f}>(.*?)</{f}>", text, re.S)
        out[f] = m.group(1).strip() if m else ""
    out["reply"] = text.split("</think>", 1)[-1].strip() if "</think>" in text else text.strip()
    return out


def build_exemplars_and_skills(src: Path):
    exemplars, ex_texts = [], []
    skills: dict[str, dict] = {}
    files = sorted(src.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    for path in files:
        case = json.loads(path.read_text(encoding="utf-8"))
        for s in case["sessions"]:
            stage = s["session_goals"]["overall_stage"]
            for meta in s["suggest_skills"]:
                meta_name = meta["meta_skill"].split(":", 1)[0]
                for m in meta["micro_skills"]:
                    sk = skills.setdefault(m["skill_id"], {
                        "skill_id": m["skill_id"], "name": m["skill_name"],
                        "description": m["skill_description"], "when_to_use": m["when_to_use"],
                        "trigger": m["trigger"], "meta": meta_name, "stages": set(), "count": 0})
                    sk["stages"].add(stage)
                    sk["count"] += 1
            dialogue = s["session_dialogue"]
            for i, turn in enumerate(dialogue):
                if not turn["role"].lower().startswith("counselor"):
                    continue
                parsed = parse_counselor_turn(turn["text"])
                prev_client = dialogue[i - 1]["text"] if i > 0 else "（会谈开始）"
                doc = {"case": path.stem, "session": s["session_number"], "turn": i, "stage": stage,
                       "position": "开场" if i == 0 else ("结束" if i >= len(dialogue) - 2 else "中段"),
                       "client": prev_client, **parsed}
                exemplars.append(doc)
                # 来访者的话权重最高（检索时用当前用户消息去匹配），评估句帮助匹配情境
                ex_texts.append(f"{prev_client} {prev_client} {parsed['assessment']} {parsed['strategy']}")
    skill_list = list(skills.values())
    for sk in skill_list:
        sk["stages"] = sorted(sk["stages"])
        sk["theory"] = "CBT"
    # DBT / ACT 技能库由 build_skills.py 从手册抽取
    for theory in ("dbt", "act"):
        p = SKILLS_DIR / f"{theory}.json"
        if p.exists():
            skill_list.extend(json.loads(p.read_text(encoding="utf-8")))
    sk_texts = [f"{s['name']} {s['name']} {s['description']} {s['when_to_use']} {s['trigger']}" for s in skill_list]
    return BM25Index(exemplars, ex_texts), BM25Index(skill_list, sk_texts), len(files)


def _book_title(stem: str) -> str:
    """文件名末尾常带下载来源、作者等括号说明，去掉后作为书名。"""
    return re.sub(r"(\s*\([^()]*\))+\s*$", "", stem).strip() or stem


def _is_toc_or_index(text: str) -> bool:
    """目录页（大量点线）和书末索引页（大量页码）关键词密集，会抢走检索结果。"""
    if len(re.findall(r"\.{4,}|…{2,}", text)) >= 3 or re.search(r"Contents|List of|Index", text[:80]):
        return True
    digits = sum(ch.isdigit() for ch in text)
    return digits / len(text) > 0.08


def build_manuals(kb_root: Path, chunk_chars: int = 1500, overlap: int = 200):
    from pypdf import PdfReader
    docs, texts = [], []
    for theory in ("cbt", "dbt", "act"):
        for pdf in sorted((kb_root / theory).glob("*.pdf")):
            title = _book_title(pdf.stem)
            reader = PdfReader(str(pdf))
            print(f"  抽取 {theory}/{title}（{len(reader.pages)} 页）", flush=True)
            n = len(reader.pages)
            front, back = (int(n * 0.06), n - int(n * 0.04)) if n > 100 else (0, n)   # 跳过书前目录、书末索引
            for pno, page in enumerate(reader.pages, 1):
                if pno <= front or pno > back:
                    continue
                try:
                    text = page.extract_text() or ""
                except Exception:
                    continue
                text = re.sub(r"[ \t]+", " ", text).strip()
                if len(text) < 200 or _is_toc_or_index(text):   # 目录、索引、插图、扫描页跳过
                    continue
                for start in range(0, len(text), chunk_chars - overlap):
                    chunk = text[start:start + chunk_chars]
                    if len(chunk) < 200:
                        break
                    docs.append({"theory": theory, "source": title, "page": pno, "text": chunk})
                    texts.append(chunk)
    return BM25Index(docs, texts)


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    print("建范例库和技能库 ...", flush=True)
    ex, sk, n_cases = build_exemplars_and_skills(PSYCHEVAL_CBT)
    ex.save(INDEX_DIR / "exemplars.pkl")
    sk.save(INDEX_DIR / "skills.pkl")
    by_theory = {t: sum(1 for s in sk.docs if s["theory"] == t) for t in ("CBT", "DBT", "ACT")}
    print(f"  {n_cases} 个案例 → 范例 {len(ex.docs)} 句；技能 {by_theory}", flush=True)
    if "--no-manuals" not in sys.argv:
        print("建手册库 ...", flush=True)
        mn = build_manuals(KB_RAW)
        mn.save(INDEX_DIR / "manuals.pkl")
        print(f"  手册块 {len(mn.docs)} 个", flush=True)
    print("完成。", flush=True)


if __name__ == "__main__":
    main()
