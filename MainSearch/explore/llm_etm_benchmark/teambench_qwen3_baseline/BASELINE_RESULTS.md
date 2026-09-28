# BASELINE_RESULTS — TeamBench × Qwen3-8B，阶段 C 正式校准（3 任务 × 3 条件 × seed 0）

日期：2026-09-28 · 运行环境：**WSL Linux (Ubuntu)** 上的官方 harness · 官方仓库
`../teambench_ref` @ `d185aef1916fd86a9ba554d581fd256319a973af`（**未修改**）

> 口径提示（外部裁定 #1/#4/#5，`PROTOCOL_REVISION.md` §1–§2）：本文所有结论仅限于
> 「**官方 harness 下的三角色协作收益**」；**不代表** OS 强隔离 / Docker 强制隔离 / 沙箱内运行。
> 费用**只给实测输入/输出 tokens 与墙钟耗时**，不给任何单价或金额。
> `full − restricted` 代表「**完整团队系统收益**」（预算不同），**不是**排除计算量差异后的因果收益。

---

## 1. 本次运行（冻结配置，未做任何覆盖）

| 项 | 值 |
| --- | --- |
| 任务（冻结） | `GH12_click_envvar_flag`、`SPEC5_config_system`、`CROSS1_api_contract` |
| 条件（冻结） | `oracle`、`restricted`、`full` |
| seed | `0` |
| 预算 | **官方默认**：oracle 20 轮 / restricted 30 轮 / full 每阶段 20 轮 + 最多 2 次补救 —— **未传** `--max-turns`、`--max-remediation` |
| 模型 | `qwen3-8b`（`MODULE_ID`，适配器实际 model id）；三条件同一适配器实例、同一采样参数 |
| 命令 | `results/phase_c_run.sh`（= 方案 §5 阶段 C 命令形态 + WSL 环境变量） |
| 起止 | 2026-09-28 12:40:34 → 13:09:50（+08:00），墙钟 **29 分 16 秒** |
| 退出码 | 0（9/9 运行完成；`error` 字段 9 次全为 `null`；`failure_modes` 仅 SPEC5 的三次运行非空，内容是 grader 未通过的检查项名，例如 `missing_keys;defaults_wrong;…`，不是 harness 异常） |

原始证据（全部保留，可逐条复核）：

| 文件 | 内容 |
| --- | --- |
| `results/phase_c_run.log` | **原始运行日志全文**（441 行；每条 turn 的 tool_calls/done、每次运行的 PASS/FAIL + partial + 耗时 + turns、full 的 phase/remediation 轨迹、最终 ABLATION COMPLETE 汇总） |
| `results/ablation_results.json` | 官方逐任务结果（runs / metrics / per_condition） |
| `results/ablation_results.json.checkpoint.jsonl` | 官方 checkpoint（9 行，与 runs 一一对应） |
| `results/ablation_runs/<task>/<run_id>/` | 9 棵 run 树：`logs/**/turn_*.json` 共 **295** 份逐 turn 日志、`messages/dialogue.jsonl`、`submission/attestation.json`、`reports/score.json` |
| `results/phase_c_trace.jsonl` | 适配器 trace：295 条 `response`（含每次调用 tokens）、295 条 `request`、295 条 `api_response_meta`（含 `finish_reason`） |
| `results/summary.csv` | 9 行的汇总（task / condition / seed / partial / passed / turns / in-out tokens / elapsed / run_id / failure_modes / error） |
| `results/summary_build.txt` | `summary.csv` 的生成过程与一致性校验输出 |
| `results/build_summary.py` | 汇总脚本（token 归属方法见 §3 注） |

---

## 2. 9 次运行结果

| # | condition | task | partial | passed | turns | 输入 tok | 输出 tok | 耗时 s | run_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | oracle | GH12_click_envvar_flag | 0.00 | false | 13 | 24 279 | 4 787 | 85.5 | `20260928_044036_83f27e44` |
| 2 | oracle | SPEC5_config_system | 0.05 | false | 10 | 30 941 | 7 539 | 150.8 | `20260928_044205_879035f4` |
| 3 | oracle | CROSS1_api_contract | 0.00 | false | 20* | 61 310 | 1 060 | 36.6 | `20260928_044442_721c5c81` |
| 4 | restricted | GH12_click_envvar_flag | 0.00 | false | 30* | 254 244 | 11 183 | 289.9 | `20260928_044523_6560d0bd` |
| 5 | restricted | SPEC5_config_system | 0.26 | false | 4 | 4 813 | 1 186 | 139.4 | `20260928_045017_01c16f8a` |
| 6 | restricted | CROSS1_api_contract | 0.20 | false | 9 | 16 895 | 488 | 28.5 | `20260928_045240_f3849b6c` |
| 7 | full | GH12_click_envvar_flag | 0.88 | false | 78 | 361 495 | 12 425 | 298.6 | `20260928_045314_09c3e747` |
| 8 | full | SPEC5_config_system | 0.05 | false | 55 | 283 639 | 14 146 | 504.8 | `20260928_045814_6971c4bf` |
| 9 | full | CROSS1_api_contract | 0.00 | false | 76 | 167 651 | 5 127 | 182.5 | `20260928_050643_e2596bab` |

`*` = 该 loop 用满了官方默认轮次上限（oracle 20 / restricted 30），属**预算截断**，非模型自行结束；
其余运行的 loop 由模型 `DONE` 或「连续 3 轮无工具调用」终止（原始日志可逐行核对）。
full 的运行内部亦有到达 20 轮上限的阶段 loop（第 7 次的 planner 与 verifier attempt 0、第 8 次的
verifier attempt 1、第 9 次的 planner），明细见 `REVIEW_PACKAGE.md` §12.5。

**通过率：9 次里 0 次 `pass=true`**（`reports/score.json` 的 `pass` 字段；官方 `per_condition`
的 success_rate 三条件均为 0.0）。

---

## 3. 条件聚合（实测 tokens 与耗时；**不报价格**）

| condition | partial 均值 | passes | 输入 tok | 输出 tok | 总 tok | turns | 耗时 s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| oracle | 0.017 | 0/3 | 116 530 | 13 386 | 129 916 | 43 | 272.9 |
| restricted | 0.153 | 0/3 | 275 952 | 12 857 | 288 809 | 43 | 457.8 |
| full | 0.310 | 0/3 | 812 785 | 31 698 | 844 483 | 209 | 985.9 |
| **合计** | — | 0/9 | **1 205 267** | **57 941** | **1 263 208** | **295** | **1 716.6**（每运行耗时之和；墙钟 1 756 s） |

token 归属方法（可复核）：官方本 commit 的 ablation JSON **不含 token 字段**，唯一逐调用记录是适配器
trace；`AgentLoop` 每轮恰好一次 `generate_with_tools` 调用，故按每棵 run 树的逐 turn 日志条数对 trace 的
`response` 记录**顺序切片**求和，并用**工具调用多重集**逐运行比对（`summary_build.txt` 末行
`consistency: OK`）。全 9 次切片与逐 turn 日志完全一致，无未归属记录。

辅助指标（方案 §4.2）：

* 工具调用：审计口径下 **240 次**工具调用（read 44 / write 47 / run 47 / other 101，
  其中 `other` 全部是 `send_message`；另有 1 次调用的参数是 JSON 字符串而非对象，
  单列 `unparsed_args=1`，不影响判定）。适配器 trace **240/240 与逐 turn 日志逐条匹配、
  0 未匹配**，无覆盖空洞；
* 消息链路：full 的 GH12 运行写入 `messages/dialogue.jsonl` **39 条**（planner→executor 36、
  executor→verifier 3），即 Planner→Executor 与 Executor→Verifier 的消息**实际送达**；
  该 run 的 planner 在 20 轮里**每轮重发同一份 plan**（`send_message` ×20），是消息总线上的
  冗余投递，供后续诊断参考；
* verifier 反馈：该运行 verifier 写 `submission/attestation.json`（`verdict=pass`，7/7 自查项 ok），
  而官方确定性 grader 给出 **7/8 检查项、partial=0.88、pass=false**（C8 `pytest test_cli.py` 未过）
  —— 即**验证者自评与 grader 判定不一致**，本基线以 grader 为准。

---

## 4. 四条冻结门槛（方案 §4.3，判定一律以本次 9 次的官方默认预算结果为准）

| # | 门槛 | 判据 | 实测 | 判定 |
| --- | --- | --- | --- | --- |
| 1 | **链路门槛** | mock 流程成功；真实 API 完成标准 function/tool call；三角色同模型且非思考模式 | 阶段 B 已满足（`preflight/b1_mock_wsl.log`、`b2_text_connectivity.log`、`b3_function_call.log`）；阶段 C trace 再次确认：`adapter_init` = model `qwen3-8b`、`enable_thinking=false`、`seed=0`、base host `https://api.gpt.ge/v1`；295 次调用全部为结构化 tool-call 协议（`lenient_mode=False`），三条件共用同一适配器实例 | **通过** |
| 2 | **可解性门槛** | 3 任务中 **≥2 个** `oracle partial_score > 0` | oracle：GH12 0.00、SPEC5 **0.05**、CROSS1 0.00 → **1/3** | **不通过** |
| 3 | **难度门槛** | `full` 不能三任务全 0 或全 1，且 ≥1 个任务取得严格介于 0 与 1 的部分分 | full：0.88、0.05、0.00 → 既非全 0 亦非全 1；GH12 0.88 与 SPEC5 0.05 均严格介于 (0,1) | **通过** |
| 4 | **协作信号门槛** | 平均 `full − restricted ≥ 0.10`，且 **≥2/3** 任务方向为正 | 配对差：GH12 **+0.88**、SPEC5 −0.21、CROSS1 −0.20；均值 **+0.157**（≥0.10）但**仅 1/3 为正** | **不通过** |

补充事实（供判读，不改变上表判定）：

* 门槛 2 中 oracle 在 CROSS1 被官方默认 20 轮**截断**（20/20），其 0.00 含预算因素；
  oracle 在 GH12 为 **13 轮自行结束**（第 12 轮 `done=True`）仍 0.00（8 项检查全灭），
  说明本模型的单智能体能力下限确实偏低，而非单纯截断；
* 门槛 3 的「难度」在 full 上表现为**极不均匀**：唯一高分 GH12 0.88 是在用满 2 次补救
  （3 阶段 + 2 次补救、78 轮）后取得；SPEC5 5 轮规划 + 3 轮验证后即失败；
* 门槛 4 的分母：restricted 在 SPEC5/CROSS1 分别取得 0.26/0.20，**高于** full 的 0.05/0.00
  （两任务均为负向差），故方向性不成立；
* 适配器 trace 的 `finish_reason` 记录完好（295 条 `api_response_meta`），本次**未出现**
  `finish_reason=length` 之外的服务端异常，也无 400/429、工具解析失败或 grader 异常。

---

## 5. 分流（按冻结规则逐条映射，不做事后调整）

冻结规则（方案 §4.3）逐条核对：

| 冻结分支 | 是否命中 | 理由 |
| --- | --- | --- |
| 四项均通过 → 进入「跨任务经验积累」设计 | **否** | 门槛 2、4 未通过 |
| 可解性/难度通过但协作信号模糊 → 只追加事前固定的 3 个扩展任务 | **否** | 该分支的前提（可解性通过）**不成立**（1/3 < 2/3），故**不据此追加扩展任务** |
| oracle 有分但 full 全零 → 先审查工具调用、消息传递与权限适配，不直接否定 benchmark | **前提不同，但属同一族** | 观测到的是**镜像现象**：oracle 天花板 ≈ 0（0.017），而 full 在 GH12 达到全部 9 次运行的**最高分 0.88**；同样属于「条件序关系异常、结论前必须先诊断」的情形 |
| tool call 不稳定或权限隔离未落实 → 停止，不运行 9 次校准 | **否** | 工具调用稳定（240/240 匹配；仅 1 次 planner 消息的参数以字符串形式给出，已单列记录）；权限隔离按裁定 1 采用官方进程内 allow-list 路径（**非** OS/Docker 强隔离），审计未发现角色契约越界（见 §6） |
| 扩展到 6 个任务后仍无协作信号 → 停止 TeamBench 路线 | **不触发** | 本次固定 3 任务、未扩展，禁止事后换任务追正结果 |

**本次分流结论：停止推进。**

1. **不进入**「跨任务经验积累」设计（四项门槛未齐）。
2. **不追加**扩展任务：追加分支的前提是可解性通过，而本次可解性未通过；在未诊断清楚之前扩展，
   等于用更多任务掩盖 oracle 天花板塌陷。
3. 先在**现有 3 任务**上做诊断（工具调用、消息传递、权限适配、预算截断），不外推 benchmark 结论、
   也不据此否定 benchmark（与「oracle 有分但 full 全零」分支同一处理原则）：
   * oracle 在 GH12 用 13 轮自行结束却 0/8，而 restricted（仅见 brief）在同任务用满 30 轮仍 0.00，
     两者都不解该任务；full 靠 planner+executor+verifier 与 2 次补救才拿到 7/8；
   * verifier 的 `attestation.json`（pass, 7/7）与确定性 grader（7/8, fail）**不一致**，
     需在后续诊断中核对 verifier 的检查口径（这是 verifier 反馈有效性的直接证据链）；
   * full 的 209 轮 / 844 k tokens 对比 restricted 的 43 轮 / 289 k tokens：本基线的
     `full` 天然拥有更多计算量，任何 `full − restricted` 差异都只能表述为**完整团队系统收益**。
4. **禁止**：因结果更换任务子集/条件/门槛，或用运行中间结果挑选 checkpoint、重跑补分。

**不适用的结论类型（硬约束）**：本轮 3 个任务、每格 1 个 seed，**不能**支持显著性、泛化性或
ETM 有效性结论；`full − restricted` 的 +0.157 也不得写成「协作本身的因果增益」「净协作收益」。
官方 `metrics` 中的 `tni = 15.6667`（interpretation: "Teamwork fully recovers the performance gap."）
在 `s_oracle = 0` 的分母下是**退化比值**，本轮不引用、不作为结论。

---

## 6. 越权审计结论（原始输出见 `results/audit_phase_c.txt` / `audit_report_phase_c.json`）

* 命令（`results/phase_c_audit.sh`）：
  `python3 audit_privileges.py --tasks-dir ../teambench_ref/tasks --json results/audit_report_phase_c.json results/ablation_runs --trace results/phase_c_trace.jsonl`
* **退出码 1**，`OVERALL: VIOLATIONS across 9 run dir(s), 3 finding(s)`；覆盖
  9 棵 run 树 / 295 份逐 turn 日志 / 240 次工具调用，trace 交叉核对 **240/240 matched, unmatched=0**。
* 3 条 finding **全部是同一类**：`SPEC_CONTENT_LEAK`（high），目标都是 `read('config_skeleton.py')`
  —— restricted×SPEC5 1 条（`logs/restricted/turn_000.json`）、full×SPEC5 2 条
  （`logs/executor/turn_000.json` 与 `logs/executor/remediation_0/turn_000.json`），
  官方工具一律 `allowed`。
* **裁定：非角色契约越权，不触发「该次运行无效」**。证据：`config_skeleton.py` 位于该 run 自己的
  `<run>/workspace`（restricted/executor 的读根之内，官方 `read` 放行），其内容由官方任务生成器
  `generators/gen_spec5_config_system.py:926` 写入、`setup_run` 落到 workspace；被点名的行
  `"""Raised when a config value fails validation."""` 同时出现在 `tasks/SPEC5_config_system/spec.md:72`，
  即**内容来源是本角色有权读的 workspace fixture**，只是该 fixture 与 spec 文本有 2 行重叠，
  触发审计器的**内容启发式**。因此这是审计器的已知盲区（启发式无法区分「字节来自 spec」还是
  「字节来自与 spec 重叠的 workspace 文件」），不是越权。
* **已记录的残余风险未发生**：`SPEC_READ_BY_UNPRIVILEGED_ROLE` = **0**，
  `READ_OUTSIDE_ALLOWED_ROOTS` = 0，`WRITE_OUTSIDE_ALLOWED_ROOTS` = 0，
  `VERIFIER_MODIFIES_SOURCE_FILE` = 0，shell 逃逸/越根绝对路径 = 0；
  即**没有**任何 `executor`/`restricted` 请求 `/task/spec.md` 或其宿主等价路径。
* **需要审阅者确认的一点**：若对停止规则作**最字面**解读（任何 high finding ⇒ 无效），
  则被判无效的是第 5 次（restricted×SPEC5）与第 8 次（full×SPEC5）运行。本文按「越权 = 违反角色
  契约」的口径裁定为**非越权**、运行有效；该裁定及其全部依据（生成器行号、spec 行号、官方
  `allowed` 放行）已完整留档，供外部复核推翻。

---

## 7. 本轮未做 / 遗留

* 未修改官方 `teambench_ref` 任何文件；未改 grader/任务/提示词/权限实现；未 `git commit`。
* 按裁定 3，`apt` 工具链缺失不处理（本轮三条件从不进入容器）。
* 按裁定 1，**Docker 隔离**（角色容器内执行、以容器为单位的工具后端）仍未落实，
  正式 ETM 实验之前必须另行解决；本文件数字只能在**开发校准**语境使用。
* 未扩展到 6 个任务、未加 seed、未做预算匹配的对照条件（因此无因果口径）。

### 审阅者裁定（2026-09-28，外部）

**第 5 次（restricted × SPEC5）与第 8 次（full × SPEC5）运行有效，不作废。当前冻结分流「停止推进」保持不变。**

- 3 条 `SPEC_CONTENT_LEAK`（HIGH）为**审计器误报**：角色读取的是它**有权访问的 workspace 夹具**，
  **没有**读取 `spec.md`、**没有**越出允许根、**没有**修改越权文件；`enforcement=allowed` 与官方工具放行一致。
- 处置：**保留原始审计产物与 `exit 1` 不变**（作为审计器覆盖局限的证据），并在此处加入本裁定说明，
  供后续引用时区分「审计器报警」与「真实越权」。你点名的风险类别在本次 9 次运行中**均为 0**：
  `SPEC_READ_BY_UNPRIVILEGED_ROLE=0`、`READ_OUTSIDE_ALLOWED_ROOTS=0`、`WRITE_OUTSIDE_ALLOWED_ROOTS=0`、
  `VERIFIER_MODIFIES_SOURCE_FILE=0`、shell 逃逸=0。
- 影响：因未作废任何运行，四条门槛的判定输入不变 ⇒ **分流结论不变**（门槛 2 可解性与门槛 4 协作信号仍未通过）。
