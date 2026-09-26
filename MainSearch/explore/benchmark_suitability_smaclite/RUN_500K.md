# SMAClite `2s3z` QMIX 50 万步诊断

在 `D:\omp` 的 PowerShell 中运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\MainSearch\explore\benchmark_suitability_smaclite\run_qmix_500k.ps1 -Seed 0
```

这会从零连续训练 50 万环境步，不加载旧模型。每约 5 万步用**相同的 100 个评测种子**测试并保存检查点；训练循环结束后额外评测并保存真实终点模型，即使终点没有恰好落在 5 万步间隔上。模型保存于 `qmix_500k_seed0/models/`，指标保存于 `epymarl_ref/results/sacred/qmix/2s3z/` 的新运行编号。

与上次 10 万步实验相比，只新增了固定评测起点与终点评测/保存；QMIX 架构、地图、探索衰减和 CPU 环境均不变。这是单种子开发诊断，不是 ETM 实验或正式统计结论。失败或中断记录要保留。

启动前短测试已完成：Sacred run 13，seed 97，只训练至 177 步；成功执行终点评测并分别保存 35 步和 177 步模型。该短测试不作为学习效果证据。
