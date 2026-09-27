# 审查包：SMAX `2s3z` 逐身份独立训练器 + 短测（2026-09-27）

> 状态：**实现验证阶段，5M 长跑未启动**，等审查者核对代码与原始输出后再决定。
> 校验指纹（sha256 前 24 位）：`train_smax_2s3z_independent.py = a7623a6de8a1a09eeb23e84a`、
> `test_preflight.py = 2d4ecdc56603d6f9544e1726`；行数 990 / 662。

## 1. 代码位置

| 文件 | 关键位置 |
| --- | --- |
| `smax_2s3z_etm/train_smax_2s3z_independent.py` | `ScannedRNN` L110-128；`IdentityActorCritic` L131-166；`build_config` L169-216；**`make_lr_schedule`（官方分段公式）L219-234**；`make_optimizer` L239-244；`RunnerState` L250-263；`Trainer.__init__` L268-285；`init_runner`（5 套独立 params L297-300 / 5 个 opt_state L301 / 5 个隐藏槽 L303-306）；`_rollout` L321-414；`_gae` L416-438；`identity_loss` L440-467；**`_update_one`（单次优化器应用 L502-503）** L469-523；**`rollout_batch`（返回 rollout 前 `hstate_init`）L525-539**；**`apply_identity_updates`（loss 用 `hstate_init[i]`）L541-560**；`_evaluate(params, keys)` L595-715（只吃 params、argmax）；`save/load_checkpoint` L764-783；`checkpoint_dir_for`（强制 `results/`）L785-793；`parse_args`（`--tiny/--resume/--save-every-segments`）L796-845 |
| `smax_2s3z_etm/test_preflight.py` | (a) L122-170 独立更新；(b) L175-234 学习率时间轴；(c) L261-324 存档恢复；(d) L327-367 评测隔离；(e) L369-449 GRU 时序/hstate 对齐；(f) L478-501 跨进程恢复；(g) L504-575 伤害归因 vs 环境账本；(h) L579-634 rollout 内 episode 边界隐状态复位 |

## 2. 命令、配置、耗时

解释器（绝对路径）：`D:/omp/MainSearch/explore/benchmark_suitability_smax/.venv/Scripts/python.exe`，工作目录 `smax_2s3z_etm`。

| # | 命令 | 耗时 |
| --- | --- | --- |
| 1 | `python test_preflight.py`（**审查者本地独立复跑**） | **4m57.1s**，8/8 PASS |
| 1' | 同上（Agent 首次运行） | 6m06.5s，8/8 PASS |
| 2 | `python train_smax_2s3z_independent.py --tiny --save-every-segments 1 --run-name preflight_cli_a` | 105.16s（程序自报），4 updates / 2 segments |
| 3 | `python train_smax_2s3z_independent.py --tiny --save-every-segments 1 --run-name preflight_cli_b --resume results/preflight_cli_a/checkpoint_00000002.bin` | 60.96s，从 update 2 续跑 |

原始日志：**审查者复跑 = `results/verify_local/preflight_rerun.log`**；tiny 运行 = `results/preflight_cli_a/run.json`、`results/preflight_cli_b/run.json`。

`--tiny` 展开配置（取自 `results/preflight_cli_a/run.json`）：`num_envs 2 | rollout_length 8 | total_updates 4 | segment_updates 2 | ppo_epochs 2 | num_minibatches 2 | lr 0.004 | anneal_lr true | clip_eps 0.05 | ent_coef 0.01 | vf_coef 0.5 | gamma 0.99 | gae_lambda 0.95 | max_grad_norm 0.25 | gru_hidden_dim 128 | fc_dim_size 128 | obs_dim 127 | action_dim 10 | num_agents 5 | steps_per_update 4 | total_optimizer_steps 16 | lr_schedule "official_piecewise_constant" | eval_episodes 4 | eval_seed 1234 | seed 0 | params_per_identity 149,643 | unit_type_names ["stalker","stalker","zealot","zealot","zealot"]`。

## 3. 五项重点的实测证据（逐字，来自审查者复跑日志）

### 3.1 独立更新 — (a) PASS (46.6s)
```
identity 1: params max|diff|=0.000e+00 opt_state max|diff|=0.000e+00 bitwise_identical=True
identity 2: params max|diff|=0.000e+00 opt_state max|diff|=0.000e+00 bitwise_identical=True
identity 3: params max|diff|=0.000e+00 opt_state max|diff|=0.000e+00 bitwise_identical=True
identity 4: params max|diff|=0.000e+00 opt_state max|diff|=0.000e+00 bitwise_identical=True
identity 0 (updated): params max|diff|=1.587e-02 loss=0.1066
hidden/env/obs untouched by the update call: True
full update param change per identity: 1.587e-02, 1.565e-02, 1.586e-02, 1.544e-02, 1.586e-02
min pairwise max|diff| between identities after a full update: 9.749e-01
```

### 3.2 学习率时间轴 — (b) PASS (0.9s)
```
N = updates*epochs*minibatches = 4*2*2 = 16 optimizer steps; steps per PPO update = 4
analytic form: lr(n) = lr0*(1 - floor(n/4)/4)
opt_step n= 0 (update 0) schedule(count=0)=4.000000000e-03 analytic=4.000000000e-03 rel_err=0.00e+00
max relative error schedule-vs-analytic over all 16 optimizer steps: 0.000e+00
optax counter equals n at every step: True
LR constant inside each PPO update: True; drops by exactly lr0/updates=1.000000e-03 at update boundaries: True
```
（另用"恒定梯度反解实际生效 LR"交叉验证，最大相对误差 1.013e-05，量级与 float32 + Adam `eps=1e-5` 一致。）

### 3.3 存档恢复 — (c) PASS (51.8s) + (f) 跨进程 PASS
```
2 segments x 2 updates; continuous ran 4 updates, interrupted+resumed ran 2 + 2 = 4 updates
segment-1 prefix of the interrupted run equals the continuous run: True
checkpoint file preflight_resume.bin (9002702 bytes); stored meta update_count=2 segment=1; restored state equals the saved state: True
params   max|leaf diff| (continuous vs save+resume): 0.000e+00
opt state (Adam m/v) max|leaf diff|:                0.000e+00 [Adam mu: 0.000e+00 over 100 leaves; Adam nu: 0.000e+00 over 100 leaves]
optimizer step counts: resumed=[16] continuous=[16] -> equal: True
update_count: resumed=4 continuous=4 (expected 4)
hidden state max|leaf diff|: 0.000e+00; env_state identical: True; carried rng identical: True
```
CLI 跨进程：run A 与 `--resume` 的 run B 终态 `params max|leaf diff| = 0.000e+00`、`opt state = 0.000e+00`、`update_count A=B=4`。

### 3.4 评测与训练隔离 — (d) PASS (28.4s)
```
params unchanged by evaluation: True (max|diff|=0.000e+00)
opt_state (Adam m/v + count) unchanged: True (max|diff|=0.000e+00, counts=[8])
repeat evaluation bitwise identical: True
evaluation reads the current parameters (after 2 updates: won 0/4 mean_return=0.0907 vs at init won 0/4 mean_return=0.0675): True
```
结构证据：`_evaluate(params, keys)` 唯一输入是 params；评测 keys 由固定 `--eval-seed` 一次生成，与训练 rng 流分离；动作取 argmax（确定性）。

### 3.5 GRU 时序 — (e) PASS (46.5s) + (h) PASS (60.7s)
```
(e1) per-identity hidden change over 2 steps: 5.5043e-01 … 5.7527e-01; min pairwise diff between identities: 7.1665e-01 (all distinct: True)
(e2) the hstate the trainer hands to the loss is the pre-roll-out hidden state: True (max|leaf diff| vs runner.hidden=0.000e+00)
     update with the roll-out-start hidden state reproduces the trainer's update bitwise: True (max|leaf diff|=0.000e+00)
(h) roll-out length 110 > max_steps 100 -> identity 2: per-unit reset flags inside the roll-out = 103 (global done flags = 4)
     loss replay from the pre-roll-out hidden state reproduces the trainer's update bitwise across that reset: True (max|leaf diff|=0.000e+00)
     using the post-roll-out hidden state instead moves the parameters differently: True (max|leaf diff|=2.020e-02)
```

### 3.6 伤害归因 — (g) PASS
```
episode 0: enemy raw health loss (all causes)=212.00, attributed damage dealt by the allies=212.00 (gap=0.00)
per-ally dealt=[66.5, 91.0, 16.0, 38.5, 0.0] taken=[160.0, 152.0, 150.0, 150.0, 0.0]
```
（用 numpy 独立复算归因规则，与血量账本精确相等；墙死无射手故不计入任何人。）

## 4. 实施中抓到并修掉的真实 bug（都由"真的跑"发现，写进记录）

| # | 症状 | 修法 |
| --- | --- | --- |
| 1 | **loss 复算用了 rollout 之后的隐状态**（与官方 `init_hstate` 语义不符；preflight (e) 判 FAIL，`max|leaf diff|=2.710e-02`） | `rollout_batch` 显式返回 rollout **之前**的 `hstate_init`，`apply_identity_updates` 强制只用它 |
| 2 | `info["returned_won_episode"]` 轴理解错（它是 per-agent (5,)，vmap 后 (E,5)） | 取 `[..., 0]` |
| 3 | 逐 env 的 episode 统计量缺身份轴，GAE vmap 报 `inconsistent sizes` | 广播到身份轴 |
| 4 | minibatch 扫描轴顺序错（时间轴 vs minibatch 轴） | `swapaxes(0,1)`，与官方一致 |
| 5 | 评测里 `unit_index` 只覆盖 5 个盟友（应为 10 个单位） | 引入 `num_units = num_allies + num_enemies` |

## 5. 明确未验证 / 存疑（交审时如实列出）

1. **真实规模吞吐未测**：`--tiny`（2 envs）的 steps/s 不可外推；默认规模 preview = 64 envs × 64 rollout × 2000 updates = **8,192,000 env steps**（注意：与之前口头说的"5M"不同，需你决定是跑 2000 updates 还是缩到 ~1250 updates ≈ 5.12M）。
2. `--tiny` 评测里 `damage_dealt=0` 是**未训练策略**的性质（argmax 未进入射程），不是指标 bug —— (g) 已证明有伤害时归因精确。
3. 跨进程逐位一致在本机实测为 0.0，但依赖 XLA 归约顺序，换机器/版本可能退化；故判据取 1e-5 容差（实测为 0）。
4. 评测动作用 **argmax**（与本仓库既有约定一致），与官方 SMAX 的**采样评测**不同 ⇒ 跨实现比胜率时需注意。
5. `identity_loss` 的 advantage 按 **minibatch** 归一化（刻意对齐官方 `_loss_fn`）。
6. 官方入口 `ippo_rnn_smax.py` 无任何保存逻辑（`grep -c "save|pickle|serialization|checkpoint|json.dump"` = 0；`train()` 只 `return {"runner_state": …}`）⇒ B 轮跑完**不会留下 checkpoint**，存档完全由本训练器负责（已实测可加载续跑）。

## 6. 等你的决定

- 是否启动长跑；若启动，规模取 **2000 updates（8.19M 步）** 还是 **~1250 updates（5.12M 步）**、单种子 ✓（按你之前的意见"单种子、500 万步，只验 P2"，对应约 1250 updates）；
- 长跑启动后我会量真实吞吐与逐段 P2 指标（同身份 early→late 画像变化 vs 重复评测波动）。
