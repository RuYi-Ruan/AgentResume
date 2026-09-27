# one-hot 臂（最终 Oracle-stage 门槛）：待批准开跑（2026-09-27）

## 0. 要判定什么

把每名伙伴的 `{u50,u600,u1250}` 直接编码为**三维 one-hot**（四名伙伴合计 **12 维**，与现有两臂输入维度、参数量相同），
其余（网络结构、训练序列、预算、伙伴冻结与 argmax、随机流、种子）**完全不变**；与**已完成的 placeholder 臂**直接比较（同一批 324 配对局面）。

- **one-hot 有收益** ⇒ SMAX 可以继续，问题出在画像构造；ETM 后续负责**从交互中推断这种潜在能力阶段**；
- **one-hot 仍无收益** ⇒ 停止在 SMAX `2s3z` 上继续投入。

## 1. 改动与代码位置

| 文件 | 位置 | 内容 |
| --- | --- | --- |
| `train_observer_gate.py` | L27-33 | 三臂 docstring（`profile` / `placeholder` / `onehot`）|
| | L146 | `ARMS = ("profile","placeholder","onehot")` |
| | L150 / L155 | `Z_CHANNEL_BY_ARM` / `Z_SEMANTICS_BY_ARM`（每臂的通道与语义文本，写入 `run.json`）|
| | L426 / L442 | `onehot_z_table()`（(4,3,3)）/ `onehot_z_vector()` |
| | L558 / L1117 | z 表**由臂名派生**（placeholder=全 0、onehot=阶段 one-hot、profile=调用方表）⇒ 任何臂都不可能被喂成别臂的通道 |
| `evaluate_observer_gate.py` | L113 / L114 / L119 / L122 | `EVAL_ARMS`、`COMPARISON_PAIRS`、`PRIMARY_PAIR = onehot − placeholder`、`SWAP_ARMS`（onehot 与 profile 做 §5；placeholder 明确跳过）|
| | L512 / L517 | `--onehot-run` / `--onehot-checkpoint` |
| `test_onehot_arm.py`（新） | — | 检查 (a)–(e) |

**语义（冻结）**：one-hot 是**真实阶段标签（oracle 级编码）**，**不是**测量画像、**不是**零通道、**也不是**"与队伍无关的真实能力"；
按身份顺序 `[1,2,3,4]`、每身份 3 维、阶段顺序 `[u50,u600,u1250] → 索引 [0,1,2]`。
§5 的 one-hot stale 交换 = 把该伙伴的 one-hot 换成**其早期阶段**的 one-hot。

## 2. 证据（实测）

| 项 | 结果 |
| --- | --- |
| 我的**独立复跑** `test_onehot_arm.py` | **5/5 PASS**，70.9 秒（日志 `results/verify_local/onehot_arm_rerun2.log`；首轮遇到同签名早期崩溃，日志保留）|
| 子代理两次完整运行 | **5/5 PASS**（`results/oracle_gate/logs/onehot_test_final_pass.log`）|
| **回归**：既有 `test_oracle_preflight.py`（10 项，**未修改**）在改动后的 trainer+evaluator 上 | **10/10 PASS**，221 秒 |
| 三臂一致性 | 输入均 **139**（127+12）、参数量均 **151,179**、同 `--seed` 下初始参数**逐位相同** |
| one-hot 正确性 | 81 组合 × 4 身份 = 324 个 3 维块：每块恰一个 1、位置 == 阶段索引 ✓（示例 `[u600,u1250,u50,u600] → [0,1,0, 0,0,1, 1,0,0, 0,1,0]`）|
| 跨模块一致 | 训练器的组合→z 映射与评测器的组合枚举（`itertools.product(range(3), repeat=4)`）逐一致 ✓ |
| 其它臂不受影响 | placeholder 仍恒 0；profile 仍等于 `profiles.json` 的 `z`（该文件 sha256 `3fdcbf1f…` 未变）|

## 3. 我打算跑的命令与预估（**等批准**）

```bash
# 训练（与已完成两臂的 flags 逐字一致，仅 --arm / --run-name 不同；--seed 默认 0 ⇒ 同初始参数）
$PY train_observer_gate.py --arm onehot \
  --updates 1250 --segment-updates 50 --num-envs 64 --rollout-length 64 \
  --ppo-epochs 4 --num-minibatches 4 --lr 4e-3 --anneal-lr \
  --eval-seeds 1234,1235,1236,1237,1238 --eval-episodes 4 \
  --save-every-segments 1 --run-name og_onehot_1250x64x64

# 三臂配对评测（**必须显式传三个 run 目录**；默认指向 tiny 目录）
$PY evaluate_observer_gate.py \
  --onehot-run results/oracle_gate/og_onehot_1250x64x64 \
  --profile-run results/oracle_gate/og_profile_1250x64x64 \
  --placeholder-run results/oracle_gate/og_placeholder_1250x64x64 \
  --out results/oracle_gate/observer_gate_eval_onehot_full.json
```

- 训练 5,120,000 步；**预估 30–40 分钟**（推断：已完成两臂实测 2420 s / 1799 s，计算量相同）；
- 评测 3 臂 × 324 = 972 局 + §5 交换；**预估 1–2 分钟**；主比较 **onehot − placeholder**（bootstrap 口径不变）。

## 4. 异常退出记录（按契约 §8）

本机 JAX/Windows **早期初始化崩溃**：`test_onehot_arm.py` 共 6+1 次调用中 **3 次**异常退出（Agent 2 次、我 1 次），
签名一致（import 后即退、**exit 1**、**无 traceback**），日志全部保留。与 one-hot 代码无关（三臂共用同一构建路径，
同代码另有多次通过）。正式跑的处理约定：**保留失败日志**、必要时 `--resume <checkpoint>` 续跑（`resumes/run_resume-NNN.json` 不覆盖 `run.json`）、
**不自动反复重启**。

## 5. 判定与边界

- 判定规则见 §0；**只做这一个门槛**，不再微调三维画像、不重复多种子；
- 结论边界：固定 324 配对局面、单种子、**不作因果归因**、不把局面执行次数当独立样本量；
- 本轮仍是**信息价值探索**，**不是 ETM 有效性的正式结论**。
