# 三人 Overcooked 得分可分性小实验

目的：先检查三人协作中，过时队友认知造成的角色重复，能否在最终出汤数上形成足够明显的差距。

- `stale_model`：A 误判伙伴分工，选择送餐，形成一名供料、两名送餐。
- `updated_model`：A 正确认识伙伴分工，选择供料，形成两名供料、一名送餐。
- 两组使用相同地图、初始位置扰动和 500 步时限。

这是脚本策略下的可分性门槛，不是 ETM 方法有效性实验。

运行：

```powershell
.\.venv\Scripts\python.exe MainSearch\explore\overcooked_three_agent_score_gap\run_experiment.py
```
