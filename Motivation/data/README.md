# Data directory

此目录保留轻量实验元数据、经验库和结果汇总。原始逐 tick `.npz` 轨迹体积较大，不提交到 Git。

- `v2_E.json`：从无经验轨迹挖掘的经验库；
- `v2_results_formal.json`：2026-09-10 决策事件修复前的探索结果，存在跨 tick 重复计数风险，不可作为最终统计；
- `mlp_impression_results.json`：结构化 L0/Lk impression 条件 MLP 的按 seed 留一可行性实验；
- `m11_bc_results.json`：PyTorch 宏观意图策略的 M11 行为克隆/GPU 链路验证；只用历史 L0 脚本轨迹，不代表训练导致的能力提升；
- `m12_closed_loop_results.json`：行为克隆 Alice 与 L0 专家在相同初始 seed 上的闭环交付对照；
- `m13_ppo_results.json`：以 M11 BC checkpoint 为 pre、仅按独立 seed 交付量选择 post 的 PPO 训练曲线；
- `m14_checkpoint_impression_results.json`：真实训练 pre/post checkpoint 上 stale/updated belief 的固定轨迹意图反事实；
- `m14_event_predictions.csv`：M14 的 3,089 个 post 测试决策事件明细，可逐行复核真实意图、三种预测及重解释标记；
- `v2_naive/*.json`：无经验采集批次的局级元数据；
- `twopot_v1/**/*.json`：历史确定性状态机实验结果；
- `llm_pilot_matrix.json`、`v2_results*.json`：历史或调试阶段汇总。

使用当前 `scripts/m9_formal.py` 重跑后，每局汇总会包含：

- `alice_decisions`：Alice 的真实决策事件数；
- `guessed_decisions`：Bob 给出合法猜测并成功按 ID 配对的事件数；
- `acc`：配对决策上的猜意准确率；
- `cause_events`：配对事件中 Alice 明确采用经验的次数；
- `cause_acc`：上述经验采用事件的猜意准确率。
