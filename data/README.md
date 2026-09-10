# Data directory

此目录保留轻量实验元数据、经验库和结果汇总。原始逐 tick `.npz` 轨迹体积较大，不提交到 Git。

- `v2_E.json`：从无经验轨迹挖掘的经验库；
- `v2_results_formal.json`：2026-09-10 决策事件修复前的探索结果，存在跨 tick 重复计数风险，不可作为最终统计；
- `v2_naive/*.json`：无经验采集批次的局级元数据；
- `twopot_v1/**/*.json`：历史确定性状态机实验结果；
- `llm_pilot_matrix.json`、`v2_results*.json`：历史或调试阶段汇总。

使用当前 `scripts/m9_formal.py` 重跑后，每局汇总会包含：

- `alice_decisions`：Alice 的真实决策事件数；
- `guessed_decisions`：Bob 给出合法猜测并成功按 ID 配对的事件数；
- `acc`：配对决策上的猜意准确率；
- `cause_events`：配对事件中 Alice 明确采用经验的次数；
- `cause_acc`：上述经验采用事件的猜意准确率。
