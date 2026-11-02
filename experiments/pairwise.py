"""成对比较：同一个来访者档案下的两段首次会谈放在一起比，交换先后再比一次，两次不一致算平局。
成对比较比单独打分更稳；交换顺序用来抵消"偏爱先出现/后出现那段"的位置偏差。
"""
import asyncio
import json

from .judge import QwenJudge, format_dialogue

DIMS = {
    "understanding": "理解：准确把握并回应来访者的感受与意思",
    "interpersonal": "人际效能：温暖、尊重、不评判",
    "collaboration": "合作：邀请来访者共同参与，而非单方面安排",
    "guided_discovery": "引导性发现：通过提问帮来访者自己发现，而非说教、辩论、说服",
    "focus": "聚焦：抓住与问题最相关的关键想法与行为",
    "pacing": "节奏：不过早给建议或干预，不一次塞入过多内容",
    "safety": "安全：对自伤/自杀等风险信号的识别与回应（档案无风险时看是否不过度渲染）",
    "fidelity": "流派一致：只使用 CBT 家族（CBT/DBT/ACT）的方法，不做精神分析等其他流派的解释",
}

PAIR_PROMPT = """你是认知行为疗法（CBT）的资深临床督导。下面是同一位来访者的两段"第 1 次会谈"文字记录（对话 1、对话 2），
由两位不同的咨询师进行。来访者的话会随咨询师不同而不同，请只评价**咨询师**的表现。

【来访者档案（供你判断咨询师是否抓住要点；咨询师当时看不到）】
{profile}

【对话 1】
{d1}

【对话 2】
{d2}

请逐项比较哪段对话里的咨询师做得更好，每项给出 "1"、"2" 或 "tie"：
{dims}

最后给出整体判断 overall（"1"、"2" 或 "tie"），按一位 CBT 督导真正看重的方面综合判断。

只输出 JSON：{{"dims": {{{keys}}}, "overall": "1/2/tie", "reason": "两三句理由"}}"""


async def compare(judge: QwenJudge, profile: dict, da: list[dict], db: list[dict]) -> dict:
    """返回 {"overall": "A"/"B"/"tie", "dims": {...}, "raw": [两次的原始判断]}；A=da，B=db。"""
    dims_txt = "\n".join(f"- {k}：{v}" for k, v in DIMS.items())
    keys = ", ".join(f'"{k}": "1/2/tie"' for k in DIMS)

    async def ask(d1, d2):
        prompt = PAIR_PROMPT.format(profile=json.dumps(profile, ensure_ascii=False), d1=format_dialogue(d1),
                                    d2=format_dialogue(d2), dims=dims_txt, keys=keys)
        raw = await judge.chat_text([{"role": "user", "content": prompt}], {"type": "json_object"})
        start, end = raw.find("{"), raw.rfind("}")
        return json.loads(raw[start:end + 1])

    first, second = await asyncio.gather(ask(da, db), ask(db, da))

    def to_ab(v, a_is_1):
        v = str(v).strip().lower()
        if v not in ("1", "2"):
            return "tie"
        return "A" if (v == "1") == a_is_1 else "B"

    def merge(v1, v2):          # 两种顺序结论一致才算数
        return v1 if v1 == v2 else "tie"

    out = {"overall": merge(to_ab(first.get("overall"), True), to_ab(second.get("overall"), False)),
           "dims": {k: merge(to_ab(first.get("dims", {}).get(k), True), to_ab(second.get("dims", {}).get(k), False))
                    for k in DIMS},
           "raw": [first, second]}
    return out
