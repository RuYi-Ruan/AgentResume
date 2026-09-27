# ObserverGate 修复轮：问题描述与待审代码（2026-09-27）

> 结论先行：**四处勘误的修改已写入代码，但我没能完成独立验证** —— 扩展后的短测
> `test_oracle_preflight.py` 在我的机器上**连续 3 次异常退出**（exit 1、无 traceback），
> 而子代理自己的日志显示 **10/10 PASS**。按要求停在这里，交你审核代码与现象。

## 1. 现象与证据（日志全部保留，未覆盖）

| # | 命令 | 结果 | 日志 |
| --- | --- | --- | --- |
| A | `python -u test_oracle_preflight.py` | exit 1；**check 1 PASS 之后、check 2 之前**死亡，无 traceback | `results/verify_local/oracle_preflight_rerun_v3.log`（1,312 B）|
| B | 同 A（重跑 1） | exit 1；**import 之后立刻**死亡，仅有 "Submoduled environments…" 一行 | `results/verify_local/oracle_preflight_retry1.log`（91 B）|
| C | 同 A（重跑 2） | exit 1；**死在 check 1 输出中途**（最后一行是 check 1 的反事实说明）| `results/verify_local/oracle_preflight_retry2.log`（1,269 B）|
| D | 子代理自己的完整运行 | **10/10 PASS** ✓（含 check 10：`max_steps=100 → eval_step_limit=101`，真实一局 77 步正常终止）| `results/verify_local/preflight_fixes_final.log`（11.9 KB）|
| E | 上一版**六项**短测（我独立复跑） | **6/6 PASS**，114.7 秒 ✓ | `results/verify_local/oracle_preflight_rerun.log` |
| F | `-X faulthandler` 诊断 | **按你要求中途停止**（无输出保留）| — |

**关键矛盾**：同一份代码，D 成功（10/10）、A/B/C 三次失败；失败点**每次不同**（import 后 / check 1 中 / check 1 后）。

## 2. 我的假设（均未确认，按可信度排序）

1. **第二个 trainer/env 的额外编译触发本机 JAX/Windows 原生崩溃。** check 2 为了造出「两臂局长不同」会另建一个不同
   `max_steps` 的 env/trainer（`test_oracle_preflight.py:216-277`）⇒ 额外的参数初始化/编译 ⇒ 撞上此前
   `ReferenceTeamEval` 定位过的同一类崩溃（MLIR lowering、`0xC0000005`、无 traceback，探针 11 次中 4 次）。
   **反证**：原生崩溃通常给出 Windows 状态码（`3221225477` / `2147483649`），而这里三次都是 **exit 1**，更像 Python 层的
   提前退出（`SystemExit(1)` / `main()` 返回 1）但它该打印的输出也没出现 ⇒ 存疑。
2. **`main()` 的失败计数/退出路径吞掉了输出。** 若 `Reporter` 计数到失败就返回 1，本该打印 `FAIL …`，
   但 A/C 的日志里连 check 1 的 `PASS` 行都没写全 ⇒ 也许输出在硬终止时丢失。
3. **内存/资源竞争**（check 7 的诊断开销量变大）⇒ 被系统/驱动杀掉。与 exit 1 也不太吻合。

## 3. 需要你审的代码位置

### 3.1 `test_oracle_preflight.py`（症状所在）
| 行 | 内容 | 为何要看 |
| --- | --- | --- |
| `858-906` | `main()`：如何跑 check 1–10、如何统计失败、返回码是什么 | 判断 exit 1 是"某检查失败"还是"进程被杀"；确认是否有 `SystemExit`/提前 return |
| `216-277` | `check_2`：**另建不同 `max_steps` 的 trainer/env** 来造两臂不同局长 | 是否产生第二次参数初始化/编译（假设 1）；能否改为复用同一 trainer、只改局长配置而不重新 init 参数 |
| `551-642` | `check_7`：跨 rollout 的 key 对齐（新加，正是我三次死亡点附近）| 该检查是否新增了额外的长 rollout/大数组 |
| `94-151`、`526-550` | `episode_records` / `observed_episode_lengths` / `rollout_bundle` / `rollout_diagnostics` / `expected_reset_obs` / `key_records` | 诊断数据的收集方式（是否在 jit 外收集大数组、是否 flush 输出）|

### 3.2 `train_observer_gate.py`（勘误 1 与流程 5）
| 行 | 内容 |
| --- | --- |
| `532-575` | `_rollout`；**勘误 1 的落点**：`step_index = ep_length`（持久局内步数，不再用 rollout 内 `step`）|
| `190-205` | `env_step_key` / `partner_action_key`（现在由 `step_index` 驱动）|
| `454-460` | `horizon` 与 `eval_step_limit = max_steps + 1`（因为环境在自增前判终止）|
| `_battle_episode` / `eval_battles` | 用 `jax.lax.while_loop` 跑到真实终止 + 截断检测（`truncated` 标志 ⇒ SystemExit）|
| `865-880` | `resume_metadata_path`：`--resume` 不再重写 `run.json`，改写 `resumes/run_resume-NNN.json` |

### 3.3 `evaluate_observer_gate.py`（勘误 3）
| 行 | 内容 |
| --- | --- |
| `330-342` | `default_matched_combo`：默认组合里被检视伙伴**真的使用 latest 阶段** |
| `343-375` | `resolve_swap_target`：返回 `(identity, param_stage, profile_stage)`，并保证 **param_stage > profile_stage**（stale，不是 future）；全 u50 组合被**拒绝**而不是翻成 future |
| `237-330` | `collect_matched_states`：匹配状态收集（伙伴已改 argmax）|
| `576-593` | 评测的终止/截断汇报（`terminated=N/total` + 长度直方图 + `battle_termination`）|

## 4. 请你判断的具体问题

1. **A/B/C 的 exit 1 到底是哪一类**：Python 层提前退出，还是本机原生崩溃被 shell 映射成 1？（我在 D 中见到 10/10，说明代码路径本身能跑通。）
2. **check 2 是否可以避免第二次参数初始化**（复用同一 trainer、只改局长配置/注入不同的重置策略）——若能，既省事也躲开假设 1。
3. **勘误 1 的正确性**：`step_index = ep_length` 与"重置注入"的交互是否会出现同一 (slot, ep, t) 跨 rollout 漂移；`ep_length` 在注入新局时是否正确归零。
4. **终止/截断守卫**：`eval_step_limit = max_steps + 1` 是否会漏掉"同时被全灭且到时限"的边界。

## 5. 状态与边界

- **两臂 5.12M 长跑未启动** ✓（预算不变 ✓）；无新实验分支 ✓；未 git commit 这三个修复文件（等你审）✓。
- 已知运行风险（记录在案，不写成"重跑即可"）：本机 JAX/Windows 参数初始化编译偶发原生崩溃。
