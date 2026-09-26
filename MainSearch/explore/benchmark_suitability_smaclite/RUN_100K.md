# SMAClite `2s3z` QMIX 训练诊断

在 PowerShell 中，从 `D:\omp` 执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\MainSearch\explore\benchmark_suitability_smaclite\run_qmix_100k.ps1 -CheckOnly
powershell -NoProfile -ExecutionPolicy Bypass -File .\MainSearch\explore\benchmark_suitability_smaclite\run_qmix_100k.ps1 -Seed 0
```

脚本使用已安装依赖的 `agentresume` Conda 环境，不调用系统默认 Python，也不额外安装依赖。该环境的 PyTorch 为 CPU 版；本机 MX450 仅有 2 GB 显存，先按已验证的 CPU 路线训练。

这是**从零开始的一次连续训练**，不是从旧的 2 万步模型续训。使用 EPyMARL 自带 QMIX 配置：探索率在 5 万步内由 1.0 降至 0.05；总预算 10 万环境步。约每 2 万步以 100 局测试并保存模型。测试局面由已记录的评测重置逻辑逐局设种子；训练重置逻辑不变。

模型保存于本目录下 `qmix_100k_seed0/models/`，Sacred 配置与指标保存于 `epymarl_ref/results/sacred/qmix/2s3z/` 的新运行编号中。实际环境步数可能略超 10 万，因为训练按完整对局结束。若中途报错，保留错误输出和已生成的运行目录，不要删除或覆盖；发给我后再诊断。

Windows 启动需使用 Sacred 的 `-C sys`：参考框架强制采用的 `fd` 捕获在本机触发 `WinError 1`。首次 10 万步尝试在环境创建前失败，记录为 Sacred run 10；`-C sys` 的 1 步预算启动测试已成功完成，记录为 run 11。二者都不是 10 万步训练结果。

这次是单种子训练门槛检查；即使出现获胜，也不能当作 ETM 有效或正式统计结论。

实际 run 12 在 100,028 步结束，但最后一次保存和独立评测停在 80,185 步；框架不会自动在训练终点再测或保存。后续延长训练前需修正这一点，不能把结束时的控制台汇总指标当作终点模型指标。
