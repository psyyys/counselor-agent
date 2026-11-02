"""换评委、对已有对话重新评分（对话不受评委影响，不必重跑会谈）。

    python -X utf8 -m experiments.rescore --sources 20261021_184349:baseline,reference 20261021_185628:counselor_a

三部分：
1. 绝对评分：PsychEval 9 项指标，评委 judge_model，加长度偏差提示
2. 成对比较：judge_model 和 judge_model_2 两个评委，每对比较两次（交换先后），不一致算平局
3. 长度敏感性检验：给 PsychEval 原对话每句咨询师发言加一段没有内容的客套话，
   看评委（有/无长度提示）会不会给"加长版"更高分——会，就说明评委偏爱长度
"""
import argparse
import asyncio
import json
from collections import Counter
from datetime import datetime

from counselor.config import load_config
from counselor.safety import HOTLINE_TEXT

from .judge import QwenJudge, evaluate, load_methods
from .pairwise import DIMS, compare
from .run import RESULTS, RISK_ASK, RISK_IN_PROFILE

FILLER = "我能感受到你说这些时心里的不容易，也很谢谢你愿意这样坦诚地告诉我。你的感受都是可以理解的，我们可以按你的节奏慢慢来。"
PAIRS = [("counselor_a", "baseline"), ("counselor_a", "reference"), ("baseline", "reference")]
LEN_METRICS = ("CTRS", "WAI", "HTAIS", "Custom_Dim")


def load(sources):
    data = {}          # (cid, cond) -> dialogue；profiles[cid]
    profiles = {}
    for item in sources:
        d, conds = item.split(":")
        for cond in conds.split(","):
            for p in sorted((RESULTS / d).glob(f"case*_{cond}.json")):
                j = json.loads(p.read_text(encoding="utf-8"))
                cid = p.name[4:].split("_")[0]
                data[(cid, cond)] = j["sessions"][0]["session_dialogue"]
                profiles[cid] = j["client_info"]
    return data, profiles


def padded(dialogue):
    return [{**t, "text": t["text"] + FILLER} if t["role"] == "counselor" else t for t in dialogue]


async def main_async(args):
    cfg = load_config()
    data, profiles = load(args.sources)
    cids = sorted(profiles, key=int)
    conds = [c for c in ("baseline", "counselor_a", "reference") if any(k[1] == c for k in data)]
    j1 = QwenJudge(cfg, cfg["judge_model"])
    j2 = QwenJudge(cfg, cfg["judge_model_2"])
    j1_raw = QwenJudge(cfg, cfg["judge_model"], length_control=False)
    methods = load_methods()
    len_methods = [m for m in methods if m.get_name() in LEN_METRICS]

    print(f"案例 {cids}，组 {conds}；评委 {cfg['judge_model']} / {cfg['judge_model_2']}", flush=True)
    abs_keys = list(data)
    pair_keys = [(cid, a, b, jn) for cid in cids for a, b in PAIRS if (cid, a) in data and (cid, b) in data
                 for jn in ("j1", "j2")]
    len_keys = [(cid, var) for cid in cids if (cid, "reference") in data
                for var in ("ref_ctrl", "pad_ctrl", "ref_raw", "pad_raw")]

    def len_task(cid, var):
        d = data[(cid, "reference")]
        d = padded(d) if var.startswith("pad") else d
        return evaluate(j1 if var.endswith("ctrl") else j1_raw, d, profiles[cid], len_methods)

    tasks = ([evaluate(j1, data[k], profiles[k[0]], methods) for k in abs_keys]
             + [compare(j1 if jn == "j1" else j2, profiles[cid], data[(cid, a)], data[(cid, b)])
                for cid, a, b, jn in pair_keys]
             + [len_task(cid, var) for cid, var in len_keys])
    results = await asyncio.gather(*tasks)
    n1, n2 = len(abs_keys), len(abs_keys) + len(pair_keys)
    abs_res = dict(zip(abs_keys, results[:n1]))
    pair_res = dict(zip(pair_keys, results[n1:n2]))
    len_res = dict(zip(len_keys, results[n2:]))

    out = RESULTS / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_rescore")
    out.mkdir(parents=True)
    (out / "raw.json").write_text(json.dumps({
        "absolute": {f"{k[0]}/{k[1]}": v for k, v in abs_res.items()},
        "pairwise": {f"{k[0]}/{k[1]}_vs_{k[2]}/{k[3]}": v for k, v in pair_res.items()},
        "length_test": {f"{k[0]}/{k[1]}": v for k, v in len_res.items()},
        "usage": {"j1": j1.usage, "j2": j2.usage, "j1_raw": j1_raw.usage}}, ensure_ascii=False, indent=1), encoding="utf-8")

    L = [f"# 重新评分 {out.name}", "",
         f"对话来源：{' '.join(args.sources)}（对话未重跑）",
         f"评委 1：{cfg['judge_model']}；评委 2：{cfg['judge_model_2']}；咨询师/来访者：{cfg['main_model']}（与评委不同家）",
         "长度偏差控制：评分前加 system 提示（长度不是标准，见 experiments/judge.py LENGTH_NOTE）；成对比较交换先后各比一次，不一致算平局。", ""]

    # 1 长度敏感性检验（先看评委可不可信）
    L += ["## 1. 长度敏感性检验（PsychEval 原对话 vs 每句加客套话的加长版，评委 1）",
          f"加的话：「{FILLER}」（不含任何咨询内容）。加长版分数更高 = 评委偏爱长度。", "",
          "| 案例 | 设置 | " + " | ".join(LEN_METRICS) + " |", "|---" * (len(LEN_METRICS) + 2) + "|"]
    for cid in cids:
        for var, name in (("ref_raw", "原对话·无提示"), ("pad_raw", "加长版·无提示"), ("ref_ctrl", "原对话·有提示"), ("pad_ctrl", "加长版·有提示")):
            s = len_res.get((cid, var), {}).get("counselor", {})
            L.append(f"| {cid} | {name} | " + " | ".join(str(s.get(m, "–")) for m in LEN_METRICS) + " |")
    for ctrl, name in (("raw", "无提示"), ("ctrl", "有提示")):
        diffs = [len_res[(c, f"pad_{ctrl}")]["counselor"].get(m, 0) - len_res[(c, f"ref_{ctrl}")]["counselor"].get(m, 0)
                 for c in cids if (c, f"pad_{ctrl}") in len_res for m in LEN_METRICS]
        if diffs:
            L.append(f"\n{name}：加长版 − 原对话 平均 {sum(diffs) / len(diffs):+.2f} 分（{sum(d > 0 for d in diffs)}/{len(diffs)} 项加长版更高）")
    L.append("")

    # 2 成对比较
    L += ["## 2. 成对比较（两种顺序结论一致才算胜，否则平局）", "",
          "| 对比 | 评委 | A 胜 | 平 | B 胜 |", "|---|---|---|---|---|"]
    for a, b in PAIRS:
        for jn, jname in (("j1", cfg["judge_model"]), ("j2", cfg["judge_model_2"])):
            c = Counter(pair_res[(cid, a, b, jn)]["overall"] for cid in cids if (cid, a, b, jn) in pair_res)
            if c:
                L.append(f"| {a}(A) vs {b}(B) | {jname} | {c['A']} | {c['tie']} | {c['B']} |")
    L += ["", "分维度（两个评委、所有案例合计；A 胜 / 平 / B 胜）：", "",
          "| 对比 | " + " | ".join(DIMS) + " |", "|---" * (len(DIMS) + 1) + "|"]
    for a, b in PAIRS:
        cells = []
        for d in DIMS:
            c = Counter(v["dims"][d] for (cid, x, y, jn), v in pair_res.items() if (x, y) == (a, b))
            cells.append(f"{c['A']}/{c['tie']}/{c['B']}")
        L.append(f"| {a} vs {b} | " + " | ".join(cells) + " |")
    L.append("")

    # 3 绝对评分
    mc, mcl = ["CTRS", "WAI", "HTAIS", "Custom_Dim", "RRO"], ["SRS", "PANAS", "SCL_90", "BDI_II", "RRO"]
    for side, ms, title in (("counselor", mc, "咨询师层面"), ("client", mcl, "来访者层面（SCL-90、BDI-II 越低越好）")):
        L += [f"## 3. PsychEval 绝对评分 · {title}（评委 1，有长度提示）", "",
              "| 组 | " + " | ".join(ms) + " |", "|---" * (len(ms) + 1) + "|"]
        for cond in conds:
            vals = []
            for m in ms:
                xs = [abs_res[(cid, cond)][side][m] for cid in cids if m in abs_res.get((cid, cond), {}).get(side, {})]
                vals.append(f"{sum(xs) / len(xs):.2f}" if xs else "–")
            L.append(f"| {cond} | " + " | ".join(vals) + " |")
        L.append("")

    # 安全
    L += ["## 4. 风险评估（档案有自杀/自伤念头的案例）", ""]
    for cid in cids:
        if RISK_IN_PROFILE.search(json.dumps(profiles[cid], ensure_ascii=False)):
            for cond in conds:
                says = [t["text"].replace(HOTLINE_TEXT, "") for t in data.get((cid, cond), []) if t["role"] == "counselor"]
                L.append(f"- 案例 {cid} · {cond}：{'✅ 问了' if any(RISK_ASK.search(s) for s in says) else '❌ 没问'}")
    errs = {f"{k[0]}/{k[1]}": v["errors"] for k, v in {**abs_res, **len_res}.items() if v.get("errors")}
    if errs:
        L.append(f"\n评分出错：{json.dumps(errs, ensure_ascii=False)}")
    L.append(f"\n用量：评委1 {j1.usage}；评委2 {j2.usage}；评委1无提示 {j1_raw.usage}")
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True, help="结果目录:组1,组2")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
