# Spread-5 筛选实验（开发阶段）

目的：先检验五名固定身份、参数互不共享的智能体能否在官方 MPE2 `simple_spread_v3` 中持续学习。只有这个门槛通过，才测试跨对局伙伴历史是否给 ETM 提供额外信息及团队收益。

预设配置：`N=5`、`local_ratio=0`、`max_cycles=25`、离散原子动作。每名智能体独立 MLP actor-critic 与优化器；无规则执行器、无策略 checkpoint 切换。评估固定 64 个地图种子，使用确定性动作，与训练随机地图分离。报告团队回报、最终目标平均最近距离、全部目标覆盖局数（距离阈值 0.15，仅辅助诊断）。

执行顺序：单种子 1 万步测速；若运行正常，单种子 10 万步学习检查；若出现明确提升，再补至少 3 个训练种子并进入 ETM 对照。门槛为至少 2/3 个种子在固定评估中，最终团队回报高于初始，并且目标最近距离下降。开发结果不能作为正式结论。

历史信息门槛：在不提供伙伴动作、目标或高层意图标签的情况下，比较“当前观察”与“当前观察 + 此前对局中的定向伙伴行为摘要”对伙伴后续行为的预测；只有历史在留出对局中确有增益，再做闭环 ETM 与冻结/无 ETM 对照。

运行：`C:\Users\86134\.conda\envs\agentresume\python.exe MainSearch\explore\benchmark_suitability_spread5\run.py --steps 10000 --seed 0 --output MainSearch\explore\benchmark_suitability_spread5\speed_seed0`
