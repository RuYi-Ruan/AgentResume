# TeamBench × Qwen3-8B 小规模基线方案

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-28
- Verification Status: UNVERIFIED（尚未执行）
- Version Label: code_plan_v1

## 1. 本轮目标

只判断 TeamBench 是否适合后续三智能体 ETM 实验，不接入记忆、经验学习或 ETM。

三个角色 Planner、Executor、Verifier 全部使用同一个 `qwen3-8b`，统一关闭思考模式。需要回答：

1. Qwen3-8B 能否稳定完成 TeamBench 的工具调用与三角色流程？
2. 任务是否既非全失败，也非接近饱和？
3. `full` 三智能体相对 `restricted` 受限单智能体是否存在初步协作收益？

本轮属于开发校准，不作为论文结论。

## 2. 固定配置

| 项目 | 固定值 |
|---|---|
| 模型 | 三个角色均为 `qwen3-8b` |
| API | OpenAI-compatible；读取 `D:/omp/MainSearch/.env` |
| 配置键 | `BASE_URL`、`API_KEY`、`MODULE_ID`、`ENABLE_THINKING=false` |
| 推理模式 | 每次请求显式传递 `enable_thinking=false` |
| 温度 | `0.2`，与 TeamBench 官方默认一致 |
| seed | `0` |
| 角色隔离 | 必须使用官方 OS/Docker 权限隔离，不接受仅靠提示词模拟 |
| 评分 | 只采用官方确定性 grader 的 `partial_score` 与 `passed` |
| 跨任务状态 | 基线阶段每个任务完全重置，不保留角色记忆 |

禁止修改官方任务、grader、角色权限、角色提示词和评分逻辑。允许的修改仅限第三方 OpenAI-compatible 适配器、配置读取、脱敏日志和运行包装。

## 3. 任务与对照

### 3.1 无 API 冒烟任务

- `DIST1_queue_race`
- 模型：官方 `mock`
- 条件：`oracle`
- 目的：验证安装、Docker、任务复制和 grader 链路，不解释得分。

### 3.2 Qwen3-8B 首轮任务（固定，不因结果更换）

1. `GH12_click_envvar_flag`：较简单的真实代码修复，作为低难度锚点。
2. `SPEC5_config_system`：Executor 缺少完整 schema，Planner 信息应有明确价值。
3. `CROSS1_api_contract`：跨 Python/Go API 合同修复，Planner 掌握三处合同差异。

每个任务依次运行三个条件：

- `oracle`：单智能体拥有完整信息，用于检查任务对该模型是否可解。
- `restricted`：单智能体只有 Executor 信息，作为受限单体基线。
- `full`：Planner + Executor + Verifier，作为三智能体基线。

总计 `3 个任务 × 3 个条件 × 1 个 seed = 9 次`。首轮不运行其他消融。

### 3.3 事前固定的扩展任务

只有首轮结果模糊时，才追加以下三个任务；不得根据首轮得分另挑有利任务：

- `PIPE2_data_pipeline`
- `TRAP1_spec_conflict`
- `DIST1_queue_race`

扩展后总计 6 个任务，条件与 seed 不变。

## 4. 指标与门槛

### 4.1 主指标

- 每个任务的官方 `partial_score ∈ [0,1]`。
- 主比较：同一任务上的 `full - restricted` 配对差。

### 4.2 辅助指标

- `passed` 局数：报告为“通过多少个任务 / 共多少个任务”。
- 三个角色的工具调用成功率、无效工具调用数、总 turns、输入/输出 tokens、耗时。
- Planner→Executor 消息是否实际送达；Verifier 是否实际读取并反馈。

### 4.3 开发门槛

1. **链路门槛**：mock 流程成功；真实 API 完成一次标准 function/tool call；三角色均使用相同模型且非思考模式。
2. **可解性门槛**：3 个任务中至少 2 个的 `oracle partial_score > 0`。
3. **难度门槛**：`full` 不能三个任务全部为 0 或全部为 1，并且至少一个任务得到严格介于 0 和 1 的部分分。
4. **协作信号门槛**：平均 `full - restricted ≥ 0.10`，且至少 2/3 个任务方向为正。

分流规则：

- 四项均通过：进入“跨任务经验积累”设计。
- 可解性/难度通过，但协作信号模糊：只追加事前固定的 3 个扩展任务。
- `oracle` 有分但 `full` 全零：先审查工具调用、消息传递和权限适配，不直接否定 benchmark。
- tool call 不稳定或权限隔离未落实：停止，不运行 9 次校准。
- 扩展到 6 个任务后仍无协作信号：停止 TeamBench 路线，不能事后换任务追正结果。

这些门槛只用于开发分流；3 或 6 个任务不能支持显著性、泛化性或 ETM 有效性结论。

## 5. 实施顺序

### 阶段 A：环境与适配器（编码 Agent 执行）

1. 将官方仓库固定在：
   `D:/omp/MainSearch/explore/llm_etm_benchmark/teambench_ref`
2. 记录 Git commit hash，安装官方依赖并完成 `docker compose build`。
3. 审计当前版本的 adapter 与 CLI。官方 README 宣称支持 `--base-url`，但实现前必须以实际代码为准；如 CLI 未暴露该参数，只做最小适配，不改 benchmark 语义。
4. 适配器从 `D:/omp/MainSearch/.env` 读取配置，不复制或打印 API Key。
5. 所有请求显式传入 `enable_thinking=false`。
6. 保持官方 function/tool-call 协议；禁止退化成从普通文本中正则猜测工具调用。

### 阶段 B：短测与审查包（暂不跑 9 次）

按顺序执行：

1. mock 冒烟测试。
2. 纯文本连通测试，期望准确返回 `API_OK`。
3. 单个虚拟工具的两轮 function-call 测试：模型发出工具调用→本地返回工具结果→模型结束。
4. 单任务、单条件、低 turns 的真实 harness 短测，只验证日志、工具调用、grader 和输出结构。
5. 检查 Docker/挂载权限确实隔离 Planner、Executor、Verifier。

出现崩溃、400/429、工具调用解析失败或 grader 异常时，保留原始失败记录并停止；不得静默重试。

编码 Agent 先提交以下审查包，等待审核后才能启动正式 9 次：

- 仓库 commit hash 与全部修改 diff。
- 完整运行命令。
- mock、API、function-call、harness 短测原始日志。
- 脱敏后的实际请求配置：模型、温度、`enable_thinking`、max tokens、base host；不得包含 Key。
- 证明三个角色模型 ID 和采样参数相同。
- 证明 `tasks/`、grader、角色权限和官方提示词未被修改。
- 预计单次与 9 次运行的 tokens、时间和费用。

### 阶段 C：正式校准（审核通过后执行）

期望命令形态如下，具体 `<adapter-model-id>` 由实现后的适配器决定：

```powershell
python -m harness.ablation `
  --model <adapter-model-id> `
  --tasks GH12_click_envvar_flag SPEC5_config_system CROSS1_api_contract `
  --seeds 0 `
  --conditions oracle restricted full `
  --output <results-dir>/ablation_results.json
```

禁止使用运行中间结果选择 checkpoint、修改任务、调整提示词或改变条件预算。

## 6. 输出目录与交付物

统一输出到：

`D:/omp/MainSearch/explore/llm_etm_benchmark/teambench_qwen3_baseline/`

至少包含：

| 文件 | 内容 |
|---|---|
| `IMPLEMENTATION_NOTES.md` | commit、依赖、改动和命令 |
| `preflight/` | 所有短测原始日志 |
| `results/ablation_results.json` | 官方逐任务结果 |
| `results/summary.csv` | task、condition、partial score、passed、tokens、turns、耗时 |
| `REVIEW_PACKAGE.md` | 代码审查证据与已知异常 |
| `BASELINE_RESULTS.md` | 按冻结门槛做出的最终分流 |

原始响应日志可保留，但必须删除鉴权头、API Key 和不必要的完整环境变量。

## 7. 本轮不做

- 不加入 ETM、长期记忆、反思模块或世界模型。
- 不让三个角色使用不同模型。
- 不比较思考模式。
- 不根据结果修改任务子集或开发门槛。
- 不把本轮 3/6 个任务的结果写成正式论文结论。

## 8. 官方依据

- TeamBench README：三角色采用 OS 强制权限隔离，每个任务使用确定性 grader，并提供 `partial_score`：https://github.com/ybkim95/TeamBench
- TeamBench 论文：团队收益具有任务难度依赖性，单体较弱时更可能出现团队收益：https://arxiv.org/abs/2605.07073
- 官方 90 任务清单与难度：https://github.com/ybkim95/TeamBench/blob/main/leaderboard/data/leaderboard_90_tasks.json

