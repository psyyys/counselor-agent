"""接入 PsychEval 的评测指标（third_party/PsychEval/eval/methods）。

PsychEval 的指标类调用 gpt_api.chat_text(messages, response_format)，并要求严格的 json_schema 输出。
千问的兼容接口用 json_object，这里写一个适配器替代它的 GPT5ChatClient。
注意：PsychEval 论文用 DeepSeek-V3.1 当评委，我们用千问，分数只能在本实验内部比较，不能直接和论文数字比。
"""
import asyncio
import json
import sys

from openai import AsyncOpenAI

from counselor.config import ROOT

EVAL_DIR = ROOT / "third_party" / "PsychEval" / "eval"


def load_methods():
    if str(EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(EVAL_DIR))
    from methods import BDI_II, CTRS, HTAIS, PANAS, RRO, SCL_90, SRS, WAI, Custom_Dim
    # 与 PsychEval eval/manager/evaluation_multi.py main() 中注册的指标相同
    return [HTAIS(), RRO(), WAI(), Custom_Dim(), CTRS(), PANAS(), SCL_90(), SRS(), BDI_II()]


# 长度偏差控制（本项目改编，PsychEval 原提示词没有这一段）：作为 system 消息加在每条评分请求前面
LENGTH_NOTE = """评分时请注意，回复的长短本身不是评价标准：
- 不要因为咨询师说得更多、讲解更全面、共情话语更多就给更高分；
- 简短但准确、恰当、推动了会谈的回复，不应因为短而扣分；
- 冗长、说教、一次塞入多个问题或多项任务，按评估标准中相应的不足处理；
- 只依据评估标准本身，看咨询师做了什么、做得是否恰当。"""


class QwenJudge:
    """名字沿用；实际可以是接口上的任何模型。length_control=True 时在每条评分请求前加 LENGTH_NOTE。"""

    def __init__(self, cfg: dict, model: str, concurrency: int = 6, length_control: bool = True):
        self.client = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=600)
        self.model = model
        self.sem = asyncio.Semaphore(concurrency)
        self.extra_body = {"enable_thinking": False}
        self.length_control = length_control
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    async def _create(self, msgs, kwargs):
        try:
            return await self.client.chat.completions.create(model=self.model, messages=msgs, temperature=0,
                                                             extra_body=self.extra_body, **kwargs)
        except Exception as e:
            # 有的模型（如 glm-5.3）不接受 enable_thinking 参数：去掉后重试，并记住
            if self.extra_body and "invalid_parameter" in str(e):
                self.extra_body = {}
                return await self._create(msgs, kwargs)
            raise

    async def chat_text(self, messages, response_format=None) -> str:
        msgs = list(messages)
        if self.length_control:
            msgs = [{"role": "system", "content": LENGTH_NOTE}] + msgs
        kwargs = {}
        if response_format:
            schema = response_format.get("json_schema", {}).get("schema")
            if schema:
                msgs = msgs + [{"role": "user", "content": "请严格只输出一个符合以下 JSON Schema 的 JSON 对象：\n"
                                + json.dumps(schema, ensure_ascii=False)}]
            kwargs["response_format"] = {"type": "json_object"}
        async with self.sem:
            for attempt in range(3):
                try:
                    r = await self._create(msgs, kwargs)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(3 * (attempt + 1))
        self.usage["calls"] += 1
        if r.usage:
            self.usage["prompt_tokens"] += r.usage.prompt_tokens
            self.usage["completion_tokens"] += r.usage.completion_tokens
        return r.choices[0].message.content or ""


def format_dialogue(dialogue: list[dict]) -> str:
    """与 PsychEval EvaluationManager.process_session 相同：client: / counselor: 逐行，去掉 <think>。"""
    lines = []
    for t in dialogue:
        text = t["text"].split("</think>", 1)[-1].strip() if "</think>" in t["text"] else t["text"].strip()
        lines.append(f"{'counselor' if t['role'].lower().startswith('counselor') else 'client'}: {text}")
    return "\n".join(lines)


def format_profile(p: dict) -> dict:
    """与 PsychEval EvaluationManager.format_client_info 相同的字段。"""
    st = p.get("static_traits", {})
    return {"static_traits": {k: st.get(k, "") for k in ("age", "name", "gender", "occupation", "educational_background",
                                                          "marital_status", "family_status", "social_status",
                                                          "medical_history", "language_features")},
            "main_problem": p.get("main_problem", ""), "topic": p.get("topic", ""), "core_demands": p.get("core_demands", ""),
            "growth_experience": p.get("growth_experiences", []), "core_brief": p.get("core_beliefs", []),
            "special_situation": p.get("special_situations", [])}


async def evaluate(judge: QwenJudge, dialogue: list[dict], profile: dict, methods=None) -> dict:
    """methods 是指标实例列表；PsychEval 的指标类本身无状态，可以多个任务共用。"""
    methods = methods or load_methods()
    diag, prof = format_dialogue(dialogue), format_profile(profile)

    async def one(m):
        try:
            return m.get_name(), await m.evaluate(judge, diag, prof)
        except Exception as e:
            return m.get_name(), {"error": f"{type(e).__name__}: {e}"}

    results = {"counselor": {}, "client": {}, "errors": {}}
    for name, score in await asyncio.gather(*[one(m) for m in methods]):
        if "error" in score:
            results["errors"][name] = str(score["error"])[:300]
        for side in ("counselor", "client"):
            if side in score and isinstance(score[side], (int, float)):
                results[side][name] = round(float(score[side]), 2)
    return results
