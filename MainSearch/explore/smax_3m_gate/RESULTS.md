# SMAX `3m` 学习门槛（结果，2026-09-25）

## 结论

**本预算下未学出胜局。** 按官方 JaxMARL SMAX IPPO 超参（LR 4e-3、clip 0.05、grad-norm 0.25、128 环境×128 步、4 epoch、4 minibatch）训练 **9,994,240 环境步**（1,269 秒 / 21 分钟、8,275 环境步/秒），每约 100 万步用**固定 100 局**评测（argmax 与采样两种动作模式）：**胜率全程 0.000**，训练侧均值回报仅从 0.20 升到 0.30（伤害奖励），训练侧胜率也是 0。

这不等于“SMAX 不可学”：JaxMARL 论文的 IPPO 需要远多于 1000 万步；本项目此前的经验是 QMIX（离策略）在五单位 `2s3z` 上 30 万步后就出现 100/100 胜（但伴随 50→0→32→0→50 的跳变）。**本门槛只说明：MLP + IPPO 在本机 1000 万步内没有形成可评测的胜利能力。**

## 两轮记录（都保留）

| 运行 | 超参 | 步数 | 吞吐 | 结果 |
| --- | --- | --- | --- | --- |
| `smax_3m_seed0_20M_offspeclr_stopped` | 误用 Overcooked 的 LR 2.5e-4 / clip 0.067（非官方） | 10,813,440（第 33/61 段时停止） | 5,300 环境步/秒 | 胜率徘徊在 0–0.06，判定为超参离规格后停止并归档 |
| `smax_3m_seed0_10M_officialhp` | 官方：LR 4e-3、clip 0.05、grad-norm 0.25、128×128 | 9,994,240 | 8,275 环境步/秒 | **胜率全程 0.000** |

## 与官方配置的差异（不冒充复现）

- 官方 `ippo_rnn_smax` 用 **GRU-128（带记忆）**，本次用纯前馈 MLP-128。SMAC 是部分可观测任务，记忆很可能是必需项——这是本轮最主要的离规格点。
- 官方 `NUM_ENVS 128 / NUM_STEPS 128 / 10M 步` 与本次一致；LR、clip、grad-norm、epoch、minibatch 均已对齐。
- 环境用 `HeuristicEnemySMAX`（脚本敌人），与本项目此前 SMAX 记录一致。

## 判定边界

- 不能据此排除 SMAX；也不能据此认为 ETM 在 SMAX 无效——训练门槛都没过，谈不上 ETM。
- 若要继续 SMAX，下一步是补上记忆（GRU，需要 BPTT）或改用离策略 QMIX（本项目唯一在这类任务上见过胜局的学习器），两者都是 2–4 小时的实现 + 训练。

## 文件

- `train_smax_gate.py`：分段训练 + 固定 100 局冻结评测（argmax/采样，胜利判据按 JaxMARL `SMAXLogWrapper` 约定：终局奖励 ≥ 1）+ 检查点。
- `results/smax_3m_seed0_10M_officialhp/run.json`、`results/smax_gate_run.log`；离规格运行归档在 `results/smax_3m_seed0_20M_offspeclr_stopped/` 与 `results/smax_gate_run_offspeclr.log`。
