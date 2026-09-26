# 三人独立 MLP 渐进学习门槛

三名智能体分别使用不共享参数的 MLP，通过每局共同获得的出汤奖励在线更新角色选择。低层移动仍由固定执行器完成。

这个实验只检查三名策略能否在连续协作中逐步形成分工，不检验 ETM。

```powershell
.\.venv\Scripts\python.exe MainSearch\explore\overcooked_three_mlp_learning_gate\run_experiment.py
```
