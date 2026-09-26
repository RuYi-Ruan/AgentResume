# 三人 MLP 集中训练、分散执行门槛

训练阶段使用团队级收益记录为三名独立 MLP 提供协调目标；执行和评估时，每名智能体只根据自己的局部输入选择角色。

该实验只验证训练路线是否比朴素独立学习稳定，不检验 ETM。

```powershell
$env:PYTHONNOUSERSITE = "1"
C:\Users\86134\.conda\envs\agentresume\python.exe MainSearch\explore\overcooked_three_mlp_ctde_gate\run_experiment.py
```
