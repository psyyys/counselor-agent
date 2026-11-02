"""模拟来访者。PsychEval 仓库没有附带来访者模拟器，这里按论文 E.6 对来访者的要求自己实现：
以档案驱动、问到才说、轻微阻抗、偶尔跑题、情绪非线性、说话风格符合 language_features、逐渐有领悟。
"""
import json

END_TAG = "[结束]"

CLIENT_SYSTEM = """你在扮演一位来访者，正在和心理咨询师进行第 {session_number} 次文字咨询。你只扮演来访者本人。

【你的完整档案（咨询师看不到，你要按它来表现，但不要照着念）】
{profile}

规则（PsychEval E.6 对来访者的要求）：
1. 说话口语化，符合档案里 language_features 描述的风格；每次 1–3 句话。
2. 逐步透露：咨询师问到相关内容才说；不要一次把档案倒出来；不使用"核心信念""认知歪曲""自动思维"等专业词。
3. 有轻微阻抗、偶尔跑题、情绪会起伏，但总体愿意合作；会谈推进后可以有一些领悟。
4. 没被问到的个人信息不要编造；档案里没有的细节可以合理地说"不太记得""说不清"。
5. 不要扮演咨询师，不要替咨询师总结。
6. 不要主动结束会谈，也不要主动提"下次再聊""今天就到这"；会谈什么时候结束由咨询师决定。
   咨询师布置了作业或提出建议，你可以回应，但会谈继续。
7. 只有当咨询师明确结束本次会谈（约好下次时间并道别）后，你才简短道别，并在最后加上 {end_tag}。"""


class ClientSimulator:
    def __init__(self, llm, model: str, profile: dict, session_number: int = 1):
        self.llm, self.model = llm, model
        self.system = CLIENT_SYSTEM.format(session_number=session_number, end_tag=END_TAG,
                                           profile=json.dumps(profile, ensure_ascii=False, indent=1))

    def reply(self, dialogue: list[dict]) -> tuple[str, bool]:
        """dialogue: [{"role": "counselor"/"client", "text": ...}]；返回 (来访者的话, 是否结束)。"""
        msgs = [{"role": "system", "content": self.system}]
        for t in dialogue:
            # 对模拟器来说，咨询师是"用户"，自己是"助手"
            msgs.append({"role": "user" if t["role"] == "counselor" else "assistant", "content": t["text"]})
        text = self.llm.chat(self.model, msgs, temperature=0.8).strip()
        ended = END_TAG in text
        return text.replace(END_TAG, "").strip(), ended
