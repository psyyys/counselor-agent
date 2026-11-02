"""语音：识别（qwen-audio-3.0-asr-flash）和朗读（qwen-audio-3.0-tts-plus）。

两个都走百炼原生接口（不是 OpenAI 兼容接口），地址由 config 里 base_url 的域名推出来。
2026-10-28 实测：识别 1.8 秒、朗读 2.9 秒；朗读一句再识别回来，一字不差。
"""
import base64
import re

import httpx

TTS_PATH = "/api/v1/services/audio/tts/SpeechSynthesizer"
ASR_PATH = "/api/v1/services/aigc/multimodal-generation/generation"

# 设置里的"声音风格"和"语速" → 朗读指令（qwen-audio 支持用自然语言控制语气、语速）。
# 用户觉得灵心偏年轻；Token Plan 里只有 tts-plus 两个系统音色，所以用指令调成熟度（2026-10-28 生成试听样本给用户听过）
STYLE_INSTRUCTIONS = {
    "natural": "语气温和、平静、亲切，像在安静的房间里和人谈心",
    "mature": "用三十五岁左右成熟女性的声音说话：音调放低一些，沉稳、从容、温和，像一位有经验的心理咨询师",
    "elder": "用四十岁左右女性温厚、平静的声音说话，声音低沉圆润，不急不缓，像可靠的长辈在耐心倾听",
    "soft": "声音放低、放轻，带一点气声，语气安静、包容、平稳，句子之间留出停顿",
}
SPEED_PHRASES = {"slow": "语速慢一些", "normal": "语速自然", "fast": "语速稍快但清楚"}

# 危机时朗读的简短版热线（屏幕上同时显示完整的热线卡片）
SPOKEN_HOTLINE = "如果你现在有伤害自己的想法，请拨打全国心理援助热线，一二三五六，紧急情况请拨打一一零或一二零。"


class Voice:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.host = cfg["base_url"].split("/compatible-mode")[0]
        self.headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
        self.client = httpx.Client(timeout=60)

    def asr(self, wav: bytes, sample_rate: int = 16000) -> dict:
        """返回 {"text", "words": [{"b", "e", "t"}]}。识别模型不返回情绪，
        但给出每个词的起止时间（毫秒），浏览器用它算语速和停顿（声音线索）。"""
        data = "data:audio/wav;base64," + base64.b64encode(wav).decode()
        body = {"model": self.cfg["asr_model"],
                "input": {"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": data}}]}]},
                "parameters": {"format": "wav", "sample_rate": str(sample_rate), "language_hints": ["zh", "en"]}}
        r = self.client.post(self.host + ASR_PATH, headers={**self.headers, "X-DashScope-SSE": "disable"}, json=body)
        if r.status_code != 200:
            raise RuntimeError(f"语音识别失败（{r.status_code}）：{r.text[:200]}")
        out = r.json().get("output", {})
        words = [{"b": w.get("begin_time"), "e": w.get("end_time"), "t": w.get("text", "")}
                 for w in (out.get("sentence") or {}).get("words") or [] if w.get("begin_time") is not None]
        return {"text": (out.get("text") or "").strip(), "words": words}

    def tts(self, text: str, voice: str | None = None, speed: str | None = None, style: str | None = None) -> bytes:
        text = clean_for_speech(text)
        if not text:
            return b""
        voice = voice if voice in self.cfg["tts_voices"] else self.cfg["tts_voice"]
        if style in STYLE_INSTRUCTIONS or speed in SPEED_PHRASES:
            instruction = f"{STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS['natural'])}，{SPEED_PHRASES.get(speed, SPEED_PHRASES['slow'])}。"
        else:
            instruction = self.cfg.get("tts_instruction")
        inp = {"text": text, "voice": voice, "format": "wav", "sample_rate": 24000}
        if instruction:
            inp["instruction"] = instruction
        r = self.client.post(self.host + TTS_PATH, headers=self.headers, json={"model": self.cfg["tts_model"], "input": inp})
        if r.status_code != 200:
            raise RuntimeError(f"语音合成失败（{r.status_code}）：{r.text[:200]}")
        url = r.json().get("output", {}).get("audio", {}).get("url")
        if not url:
            raise RuntimeError(f"语音合成没有返回音频：{r.text[:200]}")
        return self.client.get(url).content       # 链接 24 小时后失效，所以马上下载


def clean_for_speech(text: str) -> str:
    """去掉朗读时会念出来的符号：Markdown、项目符号、括号里的英文术语等。"""
    text = re.sub(r"[*#`>_]+", "", text)
    text = re.sub(r"^\s*[-•·]\s*", "", text, flags=re.M)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()
