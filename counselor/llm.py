"""模型调用。千问走 OpenAI 兼容接口；换模型只改 config.json。"""
import json
import re
import threading
import time


class LLM:
    def __init__(self, cfg: dict):
        from openai import OpenAI
        if not cfg["api_key"]:
            raise RuntimeError("没有找到 API Key：请在 config.json 里填 api_key，或设置环境变量 DASHSCOPE_API_KEY")
        self.client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=180)
        # 千问 qwen3 系列默认会先"思考"：慢一倍多、输出 token 多。<think> 格式本身已经要求逐步推理，所以默认关掉
        self.extra_body = {"enable_thinking": bool(cfg["enable_thinking"])} if cfg.get("enable_thinking") is not None else {}
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self._lock = threading.Lock()      # 抽技能、跑评测时多线程共用一个客户端

    def chat(self, model: str, messages: list[dict], temperature: float = 0.7, json_mode: bool = False) -> str:
        kwargs = {"model": model, "messages": messages, "temperature": temperature, "extra_body": self.extra_body}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        with self._lock:
            self._count(resp)
        return resp.choices[0].message.content or ""

    def _count(self, resp):
        self.usage["calls"] += 1
        if resp.usage:
            self.usage["prompt_tokens"] += resp.usage.prompt_tokens
            self.usage["completion_tokens"] += resp.usage.completion_tokens


class FakeLLM:
    """测试用：handler(model, messages, json_mode) -> str。"""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def chat(self, model, messages, temperature=0.7, json_mode=False):
        self.calls.append({"model": model, "messages": messages, "json_mode": json_mode})
        self.usage["calls"] += 1
        return self.handler(model, messages, json_mode)


def parse_json(text: str) -> dict:
    """模型偶尔会包一层 ```json，这里剥掉再解析。"""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"模型输出里没有 JSON：{text[:200]}")
    return json.loads(text[start:end + 1])
