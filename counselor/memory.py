"""长期记忆：每个用户一个文件夹，全是可读可改的 JSON。

data/users/<用户名>/
    profile.json        个案档案（PsychEval CBT 的 client_info 格式）
    plan.json           三阶段疗程计划（第 1 次会谈结束后生成）
    state.json          第几次会谈、当前阶段
    sessions/001.json   每次会谈：议程、逐字稿、总结、作业
"""
import json
import re
from datetime import datetime
from pathlib import Path

from .config import USERS_DIR

STAGES = ["问题概念化与目标设定", "核心认知与行为干预", "巩固与复发预防"]

# 15 种认知歪曲，PsychEval E.1 的固定列表（数据里用英文名）
COGNITIVE_PATTERNS = [
    "All-or-Nothing Thinking", "Overgeneralization", "Mental Filter", "Disqualifying the Positive",
    "Jumping to Conclusions", "Mind Reading", "Fortune Telling", "Magnification", "Emotional Reasoning",
    "Should Statements", "Labeling", "Personalization", "Blaming", "Catastrophizing", "Comparing and Despairing",
]


def empty_profile() -> dict:
    return {
        "static_traits": {k: "" for k in ("name", "age", "gender", "occupation", "educational_background",
                                          "marital_status", "family_status", "social_status",
                                          "medical_history", "language_features")},
        "main_problem": "", "topic": "", "core_demands": "",
        "growth_experiences": [],
        "core_beliefs": [],
        # 每项：event / conditional_assumptions / compensatory_strategies / automatic_thoughts / cognitive_pattern
        "special_situations": [],
    }


def _safe_name(user: str) -> str:
    name = re.sub(r"[^\w一-鿿-]", "_", user.strip())[:40]
    if not name:
        raise ValueError("用户名不能为空")
    return name


class UserStore:
    def __init__(self, user: str, root: Path = USERS_DIR):
        self.dir = root / _safe_name(user)
        (self.dir / "sessions").mkdir(parents=True, exist_ok=True)

    def _read(self, name: str, default):
        p = self.dir / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default

    def _write(self, name: str, data):
        p = self.dir / name
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)

    # 个案档案 / 计划 / 状态
    def profile(self) -> dict:
        return self._read("profile.json", empty_profile())

    def save_profile(self, p: dict):
        self._write("profile.json", p)

    def plan(self) -> dict | None:
        return self._read("plan.json", None)

    def save_plan(self, p: dict):
        self._write("plan.json", p)

    def state(self) -> dict:
        return self._read("state.json", {"sessions_done": 0, "stage_index": 0, "active_session": None})

    def save_state(self, s: dict):
        self._write("state.json", s)

    # 会谈记录
    def session(self, n: int) -> dict | None:
        return self._read(f"sessions/{n:03d}.json", None)

    def save_session(self, sess: dict):
        self._write(f"sessions/{sess['session_number']:03d}.json", sess)

    def new_session(self, n: int, stage: str, agenda: dict) -> dict:
        sess = {"session_number": n, "started_at": datetime.now().isoformat(timespec="seconds"),
                "stage": stage, "agenda": agenda, "transcript": [], "summary": None, "status": "active"}
        self.save_session(sess)
        return sess

    def last_finished_session(self) -> dict | None:
        n = self.state()["sessions_done"]
        return self.session(n) if n else None

    # 来访者自己的记录（心情签到、练习打卡、随手记）：只给"我的记录"板块用，不发给模型
    def records(self) -> dict:
        r = self._read("records.json", {})
        r.setdefault("moods", {})            # {"会谈号": {"before": 0-10, "after": 0-10}}
        r.setdefault("homework_done", {})    # {"会谈号-作业序号": true}
        r.setdefault("journal", [])          # [{"id", "time", "text"}]
        return r

    def save_records(self, r: dict):
        self._write("records.json", r)

    def all_summaries(self) -> list[dict]:
        out = []
        for n in range(1, self.state()["sessions_done"] + 1):
            s = self.session(n)
            if s and s.get("summary"):
                out.append({"session_number": n, "stage": s["stage"],
                            "abstract": s["summary"].get("session_summary_abstract", ""),
                            "homework": s["summary"].get("homework", [])})
        return out
