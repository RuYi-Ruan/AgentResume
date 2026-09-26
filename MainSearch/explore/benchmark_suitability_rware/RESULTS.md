# 四人 RWARE 探索结果（2026-09-24）

- 环境：原生 `rware:rware-tiny-4ag-v2`，RWARE 2.0.0，4 名智能体，每局最多 500 步；仅为开发门槛，不是正式实验。
- 环境 smoke：固定种子 0–9，停留策略 10 局共 0 次配送；随机策略 10 局共 1 次配送。原始记录见 `results/smoke.json`。
- EPyMARL 接口：4 名智能体，每人 71 维观察、5 个动作；`common_reward=True` 与 `reward_scalarisation=sum` 下，团队回报等于总配送次数。单局适配器测试配送 1 次，见 `results/adapter_smoke.json`。
- QMIX 4000 步成本探针：实际 4500 环境步，约 32 秒；第 4 局起出现梯度更新，8 局训练统计共 1 次配送，测试仍为 0。只说明训练链路可运行。
- QMIX 单种子开发试验：种子 0，从零连续训练至 100500 环境步，201 局训练，耗时 8 分 40 秒。每约 2 万步在相同的 20 个固定种子上测试一次，共 6 个检查点、120 局测试，全部 0 次配送。训练过程中有零星配送，但没有转化为贪心策略的测试收益。原始 Sacred 记录位于 `../benchmark_suitability_smaclite/epymarl_ref/results/sacred/qmix/rware_rware-tiny-4ag-v2/3/`，模型位于 `qmix_100k_seed0/models/`。运行配置：QMIX 共享智能体网络（观察附身份编号）、500 步时限、batch 32、探索率在 5 万步衰减到 0.05、每 2 万步测试 20 局。这还不是最终要求的四套固定身份独立策略。
- 当前 Conda 环境的 PyTorch 为 `2.1.0+cpu`，`torch.cuda.is_available()` 为 `False`，上述耗时均为 CPU 实测。

判断：RWARE 原生四人任务具有真实配送奖励，但此配置下 10 万步没有学出可评测策略，尚不能用于 ETM 收益试验；也不能据此断言 RWARE 不可学。按当前实测速度，直接投入多种子百万步的成本较高。下一步先核对成熟 RWARE 训练基线与更低成本的可学习设置，再决定是否继续 RWARE；不以增加步数掩盖门槛失败。
