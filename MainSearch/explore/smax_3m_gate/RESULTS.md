# SMAX 训练器自检与 2s3z 候选（2026-09-27）

## 执行顺序（外部给定，不再扩展分支）

1. `train_smax_gate.py` 修正：评测胜率统一用环境原生 `returned_won_episode`；custom 3m 加线性 LR 退火 1e-3 → 0。
2. **A 轮**：custom 3m smoke（FF MLP、10.24M 步、LR 1e-3 + anneal）——只判"能不能学"。
3. **B 轮**：官方 `ippo_rnn_smax.py` 原生 `2s3z`（GRU-128、LR 4e-3、ANNEAL_LR、默认 128 环境 × 128 步 × 10M 步）。
4. 分流：A✓B✓ → 进入 2s3z 的 fixed-identity / independent actor + P2 + Oracle；A✗B✓ → 迁移官方 RNN backbone；A✓B✗ → 查官方 config；A✗B✗ → 查环境/wrapper。

## 已完成的代码修正

- 评测胜率：`won = live & (info["returned_won_episode"][0] > 0)`（旧 `step_reward >= 1.0` 删除）。理由：SMAX 奖励含伤害/击杀等 shaped 分量，奖励阈值不等价于赢下一局。
- 线性退火：`optax.linear_schedule(config["lr"] → 0, transition_steps=--updates)` 包在 `optax.adam(...)`，由 `--anneal-lr` 控制，写入元数据。
- 归档的"贪心胜率 0.38 / 0.21"在修正口径下**作废**：降级为未验证旧结果。

## A 轮结果（custom 3m，FF MLP，LR 1e-3 + anneal，10.24M 步 / 19.4 分钟）

| 段 | 训练胜率 | 训练回报 | 评测贪心胜率 | 评测采样胜率 | 熵 | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.000 | 0.23 | 0.090 | 0.000 | — | 贪心胜率从第 1 段即固定 |
| 10 | 0.000 | 0.26 | 0.090 | 0.000 | — | 全程无上升趋势 |

**判定：A ✗**（前馈策略没能学起来；采样策略从未赢过；贪心 9% 平线不构成学习）。

## B 轮（官方原生 2s3z，默认配置）

启动于 2026-09-27 12:40 左右（`hub` 进程名 `smax_official_2s3z_default`）。
测量补丁：给官方 callback 增加一行 print（`WANDB_MODE=disabled` 时官方脚本无任何输出），**训练计算未改动**。
实测吞吐参考：64 环境配置下约 210 步/秒 ⇒ 默认 128 环境 × 10M 步为长跑（小时级）。

前几个打点（64 环境配置下的先导运行）：step 0 win 0.000 ret 0.200 → step 57,344 win 0.000 ret 0.270（entropy ~1.2、value loss 0.001、KL ~3e-4）⇒ 训练链在正常优化，但尚无胜局。
