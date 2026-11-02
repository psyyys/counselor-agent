"""读取 config.json；所有路径集中在这里。"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
KB_RAW = DATA / "kb_raw"            # 手册原文（pdf）
INDEX_DIR = DATA / "index"          # 生成的检索索引
USERS_DIR = DATA / "users"          # 每个用户的长期记忆
PSYCHEVAL_CBT = ROOT / "third_party" / "PsychEval" / "data" / "cbt"
WEB_DIR = ROOT / "web"

DEFAULTS = {
    "api_key": "",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "main_model": "qwen-plus",
    "critic_model": "qwen-flash",
    "main_temperature": 0.7,
    "critic_temperature": 0.0,
    "history_messages": 20,
    "exemplars_per_turn": 3,
    "skills_per_session": 12,
    "manual_chunks_per_session": 4,
    "max_revisions": 1,
    "enable_thinking": False,
    # 语音（打电话模式）
    "asr_model": "qwen-audio-3.0-asr-flash",
    "tts_model": "qwen-audio-3.0-tts-plus",
    "tts_voice": "longanlingxin",          # 默认音色
    # 设置里可选的音色（qwen-audio-3.0-tts-plus 的系统音色，官方音色表 2026-10-28）
    "tts_voices": {"longanlingxin": "龙安灵心 · 女声 · 知心温暖", "longanlufeng": "龙安鲁风 · 男声 · 明亮开朗"},
    "tts_instruction": "语速稍慢，语气温和、平静、亲切，像在安静的房间里和人谈心。",
    # 评测用评委。留空 = 用 main_model。换别的模型前先查价格（qwen3.8-max、glm-5.3 很贵，glm-5.3 还关不掉思考）
    "judge_model": "",
    "judge_model_2": "",
}


def load_config(path: Path | None = None) -> dict:
    cfg = dict(DEFAULTS)
    path = path or ROOT / "config.json"
    if path.exists():
        cfg.update(json.loads(path.read_text(encoding="utf-8")))
    if not cfg["api_key"] or cfg["api_key"].startswith("在这里填"):
        cfg["api_key"] = os.environ.get("DASHSCOPE_API_KEY", "")
    cfg["judge_model"] = cfg["judge_model"] or cfg["main_model"]
    cfg["judge_model_2"] = cfg["judge_model_2"] or cfg["judge_model"]
    return cfg
