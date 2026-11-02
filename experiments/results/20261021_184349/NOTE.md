# 注意
本目录中 counselor_a 组的结果**无效**：DeepSeek 接口吞掉了 <think> 标签，导致咨询师的内部思考和回复连在一起发给了来访者与评委（2026-10-21 已修复，改用 <reply> 标签）。
baseline 与 reference 两组有效，被后续实验通过 --reuse 沿用。
