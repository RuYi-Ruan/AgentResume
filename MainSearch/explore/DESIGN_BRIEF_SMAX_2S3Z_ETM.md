# SMAX `2s3z` 作为 ETM 主实验环境：设计与代码位置（待审核）

> 前置判定已完成：**A ✗**（custom FF 在 `3m` 上 10.24M 步学不起来）/ **B ✓**（官方 `ippo_rnn_smax.py` 原生
> `2s3z` 约 1.8M 步 win 0.69）⇒ 按矩阵**迁移官方 RNN PPO backbone**，不再调 custom FF，`3m` 使命结束。
> 本文只描述 `2s3z` 上的 ETM 前置 Gate 怎么搭，**未开跑**，等审核。

---

## 一、这一阶段的唯一目标

不是"把 SMAX 训得更高分"，而是回答一个问题：

> **同一个固定身份（例如 Stalker#0 vs Stalker#1）的伙伴，其随训练自然成长的能力变化，
> 是否改变 ego 的最优协作决策（并提高团队收益）？**

- 先证 **P2**：同一身份的 early→late 能力/行为确实在变（不能是单位类型差异——类型是可直读属性）。
- 再做 **Oracle Gate**：`Oracle(z_true) > NoInfo`，且行为分布出现可解释变化（target 选择、后撤、站位）。

---

## 二、环境事实（已核验）

- `2s3z` = `Scenario([2,2,3,3,3]*2, 5, 5)`（`jaxmarl/environments/smax/smax_env.py:57-65`）⇒ 5 个盟友：
  **agent0–1 = Stalker、agent2–4 = Zealot**；敌人由 `HeuristicEnemySMAX` 脚本控制。
- 观测：每个单位独立局部观测 + `avail_actions` 掩码；动作为 SMAX 离散 16 维（移动/攻击/停等组合）。
- 胜利：必须用 `SMAXLogWrapper.returned_won_episode`（已在 `smax_3m_gate/train_smax_gate.py:356-360` 修正）。

## 三、官方参考实现与需要改动之处

官方 `jaxmarl_ref/baselines/IPPO/ippo_rnn_smax.py` 的结构：

- `ScannedRNN`（L25）、`ActorCriticRNN`（L53）、`Transition`（L97）、`make_train`（L145）——**所有单位共用一套参数**
  （一个 `TrainState`，actor 头 `actor_mean*` 与 critic 头按名字切分，隐藏层/GRU 共享）。

要迁移成 ETM 要求的"固定身份 + 各自独立优化"：

| # | 改动 | 理由 |
| --- | --- | --- |
| 1 | **每个盟友身份一套完整参数**（含自己的 GRU 与隐藏层 + actor 头） | "各自独立优化"要求不共享表征；"共享躯干 + 独立头"会让身份间隐状态耦合 ✗ |
| 2 | **每个身份独立 optimizer**：scan 携带 `(params_i, opt_state_i)`，`upd, st = tx.update(...)` → `optax.apply_updates(params, upd)` | 复用 `overcooked_v2_3p_gate/train_ocv2_independent.py` 的修复模式；**绝不要** `TrainState.apply_gradients(grads=tx.update(...))`（双重优化器 ✗） |
| 3 | **逐身份 GRU 隐状态**：`hstate` 形状从官方 `(1, num_actors, hidden)` 改为 `(num_allies, num_envs, hidden)` | 独立策略各自持有记忆 |
| 4 | **CTDE critic 可共享**（输入全局状态，含敌方信息） | 与 Overcooked 侧一致：共享价值只做信用分配，不改变"各自独立优化" |
| 5 | **必须保存 checkpoint**（官方脚本什么都不存 ✗） | P2 与 Oracle 门槛都要用 early/late 参数 |
| 6 | **逐身份指标**：伤害造成/承受、存活步数、击杀数、到最近敌距离（kiting）、集火占比（同一敌人同窗口被 ≥2 盟友攻击） | P2 的判据来源 |
| 7 | 敌人固定为脚本 `HeuristicEnemySMAX`，随机种子在"同一批 env seeds"下对齐 | 配对比较要同敌情 |

## 四、交付物（三个脚本 + 复用既有工具）

1. `MainSearch/explore/smax_2s3z_etm/train_smax_2s3z_independent.py`
   —— 固定身份独立 actor（P2 载体）。默认：GRU-128、LR 4e-3 + 线性退火、clip 0.05、grad-norm 0.25、
   64–128 envs × 128 步、分段 `--segment-updates` 记录逐身份指标并落盘 `npz`。
2. `MainSearch/explore/smax_2s3z_etm/profile_capabilities_smax.py`
   —— 按身份统计能力画像（early vs late）+ 同一类型两两策略 TV / 动作分布重叠率。
3. `MainSearch/explore/smax_2s3z_etm/evaluate_oracle_gate_smax.py`
   —— 冻结伙伴（early 或 late）与观察者配对评测：`no_oracle` vs `oracle`（共同随机数、matched-state TV/JS、
   切换率），输出 Δ 与 bootstrap CI。
4. 复用：`overcooked_v2_3p_gate/evaluate_oracle_gate.py` 的配对/匹配状态评测框架、
   `overcooked_v2_3p_gate/measure_throughput.py` 的吞吐口径思路、`train_smax_gate.py` 的环境构造与胜率口径。

## 五、算力预算（本机 CPU-only，实测）

| 项 | 实测/估计 |
| --- | --- |
| 官方共享参数 GRU `2s3z` | **~1,140 步/秒**（10M 步 ≈ 2.4 h） |
| 独立 5 套 actor（推定慢 1.5–2×） | ~600–750 步/秒 ⇒ 5M 步 ≈ **2–2.3 h/臂** |
| 4 臂 × 2 seeds × 5M | **16–19 h**（可通宵，但不适合反复试错） |

**建议的首轮**：先 `1 臂 × 5M 步`（约 2 h）只回答 P2 —— 同身份 early→late 的能力/行为是否真的变化、
且幅度是否大于同一检查点内的身份间噪声。P2 不成立 ⇒ 不建 Oracle 门槛，直接收束（与 Overcooked 同样处理）。

## 六、待审核确认

1. 形态：**每身份独立整套 GRU 参数 + 共享 CTDE critic**（上表 #1–#4）是否同意？
2. 预算：首轮先 `1 臂 × 5M`（2 h，只看 P2）还是直接 `2–5 seeds × 5–10M`？
3. 能力向量 `z` 的构造：`[伤害效率, 存活步数, 集火贡献]` 三维 + 训练内分位归一化 —— 维度/归一化方式是否合适？
4. Oracle 门槛判据沿用 Overcooked 侧：配对 Δ > 0 且 bootstrap 95% CI 不含 0，**外加**行为分布变化（target/后撤/站位）
   的可解释性检查 —— 是否照此执行？
