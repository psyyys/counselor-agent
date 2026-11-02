"""本地网页后端：python -m uvicorn counselor.server:app --port 8765"""
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from .config import USERS_DIR, WEB_DIR, load_config
from .llm import LLM
from .memory import STAGES, UserStore, _safe_name
from .session import Counselor, Indexes

app = FastAPI(title="咨询师 A")
_counselor: Counselor | None = None
_voice = None


def voice():
    global _voice
    if _voice is None:
        from .voice import Voice
        cfg = load_config()
        if not cfg["api_key"]:
            raise HTTPException(400, "没有找到 API Key")
        _voice = Voice(cfg)
    return _voice


def counselor() -> Counselor:
    global _counselor
    if _counselor is None:
        cfg = load_config()
        try:
            llm = LLM(cfg)
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        _counselor = Counselor(llm, cfg, Indexes())
    return _counselor


class UserReq(BaseModel):
    user: str


class TurnReq(BaseModel):
    user: str
    text: str
    voice_cues: dict | None = None    # 语音通话时浏览器算出的声音线索：{"summary": str, "metrics": {...}}


class ProfileReq(BaseModel):
    user: str
    profile: dict


def _call(fn, *args):
    try:
        return fn(*args)
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.get("/api/stages")
def stages():
    return {"stages": STAGES}


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/avatars/{name}")
def avatar(name: str):
    p = (WEB_DIR / "avatars" / name).resolve()
    if p.parent != (WEB_DIR / "avatars").resolve() or not p.is_file():
        raise HTTPException(404, "没有这个文件")
    return FileResponse(p)


@app.post("/api/start")
def start(req: UserReq):
    return _call(counselor().start, req.user)


@app.post("/api/turn")
def turn(req: TurnReq):
    if not req.text.strip():
        raise HTTPException(400, "消息不能为空")
    return _call(counselor().turn, req.user, req.text, "", req.voice_cues)


@app.post("/api/end")
def end(req: UserReq):
    return _call(counselor().end, req.user)


@app.get("/api/memory/{user}")
def memory(user: str):
    store = UserStore(user)
    state = store.state()
    sessions = [store.session(i) for i in range(1, state["sessions_done"] + 1)]
    active = store.session(state["active_session"]) if state["active_session"] else None
    return {"state": state, "stage": STAGES[state["stage_index"]], "profile": store.profile(), "plan": store.plan(),
            "sessions": [{"session_number": s["session_number"], "stage": s["stage"],
                          "theory": s["agenda"].get("theory_select"), "title": s["agenda"].get("stage_title"),
                          "summary": s.get("summary")} for s in sessions if s],
            "active": active}


class TTSReq(BaseModel):
    text: str
    hotline: bool = False
    voice: str | None = None
    speed: str | None = None        # slow / normal / fast
    style: str | None = None        # natural / mature / elder / soft


@app.post("/api/voice/asr")
async def voice_asr(request: Request, sr: int = 16000):
    """请求体就是 wav 音频本身（浏览器录音后编码成 16kHz 单声道 wav）。"""
    wav = await request.body()
    if len(wav) < 1000:
        return {"text": ""}
    return _call(voice().asr, wav, sr)


@app.post("/api/voice/tts")
def voice_tts(req: TTSReq):
    from .voice import SPOKEN_HOTLINE
    text = req.text + ("\n" + SPOKEN_HOTLINE if req.hotline else "")
    return Response(_call(voice().tts, text, req.voice, req.speed, req.style), media_type="audio/wav")


@app.get("/api/options")
def options():
    """设置页需要的选项（音色列表等）。"""
    cfg = load_config()
    return {"voices": cfg["tts_voices"], "default_voice": cfg["tts_voice"]}


@app.get("/api/users")
def users():
    """这台电脑上已有的代号（本地单人使用），按最近会谈时间排序，给欢迎页做"继续之前的代号"。"""
    out = []
    if USERS_DIR.exists():
        for d in USERS_DIR.iterdir():
            sp = d / "state.json"
            if sp.is_file():
                st = json.loads(sp.read_text(encoding="utf-8"))
                out.append({"user": d.name, "sessions_done": st["sessions_done"], "active": bool(st["active_session"]),
                            "mtime": max(p.stat().st_mtime for p in d.rglob("*.json"))})
    return {"users": sorted(out, key=lambda u: -u["mtime"])}


@app.get("/api/peek/{user}")
def peek(user: str):
    """欢迎页输入代号时用：只读、不创建文件夹，看是不是老用户。"""
    try:
        d = USERS_DIR / _safe_name(user)
    except ValueError:
        return {"exists": False}
    if not (d / "state.json").exists():
        return {"exists": False}
    store = UserStore(user)
    st = store.state()
    last = store.last_finished_session()
    return {"exists": True, "sessions_done": st["sessions_done"], "active": bool(st["active_session"]),
            "stage_index": st["stage_index"], "stage": STAGES[st["stage_index"]],
            "last_title": last["agenda"].get("stage_title") if last else None,
            "last_ended": last.get("ended_at") if last else None}


# ---------- 我的记录（来访者自己看的板块） ----------
class MoodReq(BaseModel):
    user: str
    session_number: int
    phase: str                  # before / after
    value: int


class HomeworkReq(BaseModel):
    user: str
    session_number: int
    index: int
    done: bool


class JournalReq(BaseModel):
    user: str
    text: str = ""
    id: str | None = None       # 删除时用


@app.get("/api/records/{user}")
def records(user: str):
    store = UserStore(user)
    st, rec, prof = store.state(), store.records(), store.profile()
    n = st["active_session"] or st["sessions_done"]
    sessions = []
    for i in range(1, n + 1):
        s = store.session(i)
        if not s:
            continue
        summ = s.get("summary") or {}
        hw = summ.get("homework") or []
        sessions.append({
            "session_number": i, "status": s.get("status"), "started_at": s.get("started_at"), "ended_at": s.get("ended_at"),
            "stage": s.get("stage"), "theory": s["agenda"].get("theory_select"), "title": s["agenda"].get("stage_title"),
            "abstract": summ.get("session_summary_abstract"), "turns": sum(1 for t in s["transcript"] if t["role"] == "client"),
            "homework": [{"text": h, "done": bool(rec["homework_done"].get(f"{i}-{k}"))} for k, h in enumerate(hw)],
            "mood": rec["moods"].get(str(i), {}),
            "transcript": [{"role": t["role"], "text": t["text"], "time": t.get("time")} for t in s["transcript"]]})
    return {"stage_index": st["stage_index"], "stages": STAGES, "sessions_done": st["sessions_done"],
            "active_session": st["active_session"], "plan": store.plan(),
            "profile": {"main_problem": prof.get("main_problem"), "core_demands": prof.get("core_demands"),
                        "core_beliefs": prof.get("core_beliefs", []), "special_situations": prof.get("special_situations", [])},
            "sessions": sessions, "journal": rec["journal"]}


@app.post("/api/records/mood")
def save_mood(req: MoodReq):
    if req.phase not in ("before", "after") or not 0 <= req.value <= 10:
        raise HTTPException(400, "心情分数要在 0 到 10 之间")
    store = UserStore(req.user)
    rec = store.records()
    rec["moods"].setdefault(str(req.session_number), {})[req.phase] = req.value
    store.save_records(rec)
    return {"ok": True}


@app.post("/api/records/homework")
def save_homework(req: HomeworkReq):
    store = UserStore(req.user)
    rec = store.records()
    key = f"{req.session_number}-{req.index}"
    if req.done:
        rec["homework_done"][key] = True
    else:
        rec["homework_done"].pop(key, None)
    store.save_records(rec)
    return {"ok": True}


@app.post("/api/records/journal")
def add_journal(req: JournalReq):
    text = req.text.strip()[:2000]
    if not text:
        raise HTTPException(400, "内容不能为空")
    from datetime import datetime
    import uuid
    store = UserStore(req.user)
    rec = store.records()
    item = {"id": uuid.uuid4().hex[:10], "time": datetime.now().isoformat(timespec="seconds"), "text": text}
    rec["journal"].insert(0, item)
    store.save_records(rec)
    return item


@app.post("/api/records/journal/delete")
def delete_journal(req: JournalReq):
    store = UserStore(req.user)
    rec = store.records()
    rec["journal"] = [j for j in rec["journal"] if j["id"] != req.id]
    store.save_records(rec)
    return {"ok": True}


@app.get("/api/export/{user}")
def export(user: str):
    """导出这个代号的全部记录（档案、计划、每次会谈的逐字稿和总结）。"""
    store = UserStore(user)
    st = store.state()
    n = st["active_session"] or st["sessions_done"]
    data = {"user": user, "state": st, "profile": store.profile(), "plan": store.plan(),
            "sessions": [store.session(i) for i in range(1, n + 1)], "my_records": store.records()}
    return Response(json.dumps(data, ensure_ascii=False, indent=1), media_type="application/json",
                    headers={"Content-Disposition": "attachment; filename*=UTF-8''counselor_export.json"})


@app.put("/api/memory/profile")
def save_profile(req: ProfileReq):
    UserStore(req.user).save_profile(req.profile)
    return {"ok": True}
