# 项目交接说明（给接手的 codex / 同学）

> **当前阶段请先读：`docs/experiments/HANDOFF_2026-09-19.md`（M19 九组实验收口）**
> 它取代了本文档下方关于"M18 当前状态"的结论部分；下方的环境/依赖/代码框架说明仍然有效。
> 一句话现状（2026-09-19）：M19 九组（3 地图 × 3 种子）全部跑完并通过统计——意图准确率 旧印象 21.78% / 新印象 90.00% / 无印象 82.67%（9/9 组同向，McNemar p≈2.7e-84）；"印象与真身匹配 vs 不匹配" = 84.4% vs 16.2%；后续得分 新−旧 +0.393 碗（p<1e-4，8/9 组正向）。完整结果见 `docs/experiments/M19_RESULTS.md`。

> **历史状态（2026-09-16，已被取代）**：9 个 Alice 均通过成长测试……（下略，保留作历史记录）

> **长期沟通偏好（2026-09-16 用户明确要求）**：默认用简体中文和自然易懂的说法交流，减少学术化、专业化表达；先讲结果、当前进度和下一步。专业词必须配通俗解释。这个偏好也写入根目录 `AGENTS.md`，后续接手本项目时继续遵守。

> **M18 当前状态（2026-09-16）**：9 个 Alice 均通过成长测试；严格配对 A 为 9×50=450 个；自然 D 不硬凑，按固定100局实际保留 392 个（B 图三组为38/27/27），D-v3 已获批准；Bob-v5 在九组144个开发起点上，知道正确意图后前300步平均+0.1667份汤，已冻结用于正式团队得分。正式 Qwen 尚未产出结果，入口为 `scripts/m18_formal.py`，记录见 `docs/experiments/M18_PROGRESS.md`。

> 2026-09-11 M14/H2/H3 完成：真实 pre/post checkpoint 重新采集 8,342 个训练事件与 5,564 个独立测试事件。在同一 post 测试事件上，stale=66.49%、updated=100%、state-only=99.61%；39 个预定义重解释事件分别为 23.08%/100%/71.79%。updated-stale 在 16/16 seed 同向；重解释子集按 seed 提升 73.96pp，bootstrap 95% CI `[61.46, 87.5]`。updated 仍是 oracle belief，下一步需 probe classifier 验证 inferred 更新。结果见 `data/m14_checkpoint_impression_results.json`。

> 2026-09-11 M13/H1 完成：从 M11 BC checkpoint 用宏观 semi-MDP PPO 继续训练 30 updates，仅按 validation 交付量选择 post。独立 test seeds 3001–3016 上 pre=15.0 碗、post=24.625 碗（23–25），16/16 同向，配对差 +9.625，bootstrap 95% CI `[9.3125, 9.875]`。结果见 `data/m13_ppo_results.json`。下一步用这两个真实训练 checkpoint 做 M14 的 H2/H3 印象滞后反事实。

> 2026-09-10 M12 完成：`TrainableMacroAgent` 已接入闭环，使用固定几何目标解析和连续阻塞退让。seeds 1–16、700 tick 下 BC Alice 与 L0 专家均稳定达到 15 碗/局，结果见 `data/m12_closed_loop_results.json`。初版曾因双向占位在 11/16 局卡死，3-tick 让位冷却已修复。下一步为 M13 PPO 训练，pre 使用 M11/M12 的 BC checkpoint。

> 2026-09-10 M11 完成：新增 `src/ocres/trainable.py` 与 `scripts/m11_trainable_policy.py`，实现 92 维历史结构化观测、物理合法意图掩码、小型 PyTorch 策略、episode/seed 数据集和 checkpoint。L0 的 7-seed 训练 / seed8 测试均为 100%，仅用于验收接口和 CUDA 链路；历史轨迹高度同构，不能解释为泛化或能力成长。结果见 `data/m11_bc_results.json`，下一步为 M12 闭环行为克隆评估。

> 2026-09-10 环境更新：当前推荐使用 Conda 环境 `C:\Users\86134\.conda\envs\agentresume`，已安装 `torch==2.1.0+cu118` 并验证 NVIDIA GeForce MX450 可用。运行前设置 `PYTHONNOUSERSITE=1`，不要同时激活仓库旧 `.venv`。最新研究判断和可训练策略路线见 `docs/experiments/RESEARCH_STATUS_AND_ROADMAP.md`。

> 2026-09-10 口径修复：`LLMPlayer` 现为每次真实决策生成唯一 `decision_id`，并在下一 tick 发布；Bob 的 `guess_event` 按该 ID 与 Alice 决策精确配对。旧的 `data/v2_results_formal.json` 来自修复前的逐 tick 重复计数口径，仅作探索记录，正式结论必须重跑。

> 2026-09-10 新方向：参考 RECOLLAB 后，将“印象”明确为 Bob 对 Alice 能力/策略类型的 belief，而不是交给 MLP 解释自然语言。`scripts/m10_mlp_impression.py` 已完成结构化 impression 的离线可行性验证；设计见 `docs/experiments/experiment_design_mlp.md`。历史轨迹高度确定，下一阶段必须用真正训练得到的 pre/post Alice checkpoint 重做。

> 目标一句话：在 Overcooked 里验证一个"Agent Resume / 印象滞后"的动机命题——
> **伙伴 Alice 的能力因为积累了经验而提升后，如果另一个智能体 Bob 对 Alice 的印象还停留在过去，Bob 会系统性地误读 Alice 行为背后的意图；当 Bob 持有/更新了同一份经验，就能读得准。**

当前主攻方向：**基于 LLM 的智能体实验**（用硅基流动 SiliconFlow 的 `Qwen/Qwen3.5-9B`）。
早期还有一套"手写规则的确定性 agent"管线（组 A），已被用户明确放弃作为结论依据，代码保留作工具/对照，不要在主结论里引用它。

---

## 1. 环境与依赖

- Windows 11 + Python 3.11（当前推荐 Conda 环境 `agentresume`；仓库根 `.venv` 为旧环境）
- `overcooked-ai==1.1.0`，**必须 `numpy==1.26.4`（<2）**，另需 `scipy`（该包未声明依赖）；见 `requirements.txt`
- 运行方式（注意要设 PYTHONPATH，否则 `import ocres` 失败）：
  ```
  conda activate agentresume
  $env:PYTHONNOUSERSITE = "1"
  $env:PYTHONPATH = "src"
  python scripts/<脚本>.py [reps]
  ```
- LLM 配置：
  - key 放在仓库根 `apikey.txt`（**第一行**，请勿提交/回显）；`src/ocres/llm.py` 会自动读取
  - 端点默认 `https://api.siliconflow.cn/v1`，模型默认 `Qwen/Qwen3.5-9B`
  - 可在 `.env` 覆盖：`SILICONFLOW_API_KEY` / `QWEN_MODEL` / `QWEN_BASE_URL`
  - 请求关闭了思考：body 里带 `"thinking": {"type": "disabled"}`（否则 9B 会把 token 全花在 reasoning 上、content 为空）
  - 实测：单次调用 ~8–12 秒；一局 260–300 tick 通常 40–120 次调用，耗时 2–10 分钟

## 2. 目录与代码框架

### 2.1 环境/几何/执行层（稳定，改动需谨慎）
- `src/ocres/grid.py`：坐标/可通行格/各功能位（锅、洋葱台、盘台、出餐口）的站位与接近格；`World.make(layout=..., grid_rows=...)` 建环境。自定义布局可用 `grid_rows=[...]` 传 ASCII 行。
- `src/ocres/recipes.py`：锅状态（empty/items1-3/cooking/ready）、持有物、烹饪剩余时间等工具函数。
- `src/ocres/executor.py`：**电机层**。把"去某格"、"对着某目标交互"翻译成原子动作（4 向移动 + `interact`；朝向靠"先到接近格再迈入站位"保证）。`interact_action()` 返回下一条动作或 `None`（表示等待）。
- `src/ocres/runner.py`：逐 tick 循环 + 记录日志。agent 接口：
  ```python
  action(state, deliveries) -> (action, intent, target[, info_dict])
  ```
  日志行包含：`t, a, intent0/1, target0/1, p0/p1, held0/1, pot, r, info0, info1`。
  `info` 用于携带 Bob 的猜测（`guess`）与 Alice 的经验使用（`cause`）。
- `src/ocres/sensor.py`：**部分观测传感器**（当前主线要求）。给"以自我为中心 3×3 视野 + 自己手持 + 地图先验"，锅态只在视野内刷新，并带时间戳；`cooking` 超过 22 tick 会推断为"可能已好"。`obs_text()` 输出给 LLM 的文字。

### 2.2 LLM 智能体主线（`src/ocres/llm_player.py`）
当前唯一在跑的主线。`LLMPlayer` 的关键点：
- **只在决策点调用 LLM**（开局 / 长等待 / 目标无法推进 / 节奏上限 / 伙伴发布新决定时）。其余 tick 由电机层继续执行当前目标。
- **两者都是同一个类**：`Alice`（`bob_knows=None`）与 `Bob`（`bob_knows=False/True`）。行为差异**只来自上下文注入**：
  - Alice 的 `experiences`（经验库 E；非 Bob 时按情境检索 Top-k 注入，采纳时在 JSON 的 `cause` 里自报编号）；
  - Bob 的 `bob_knows`：`False`=不知情（只按常识猜），`True`=持有同一份 E 并被要求"模拟：如果我是带这些经验的 Alice 会怎么选"。
- **Bob 只在 Alice 发布新决定时猜**：通过共享 `board`（意图板，延迟 1 tick 可见）检测到 Alice 新条目时触发一次决策/猜测。
- **电机层护栏（非决策规则）**：
  - 取洋葱的"容量约束"：锅总剩余需求不足（或只剩最后 1 个且伙伴更近）时不允许取，防止两人过量持洋葱互相锁死；
  - 持洋葱但无锅可放时，停到自由格 `(3,2)` 而不是站在锅位堵住取汤；
  - 非对称退让：同一动作在同一位置连续被挡（me0≥3 / me1≥4 次）就让一步；
  - 反自旋：连续两次决策同一目标且位置没动，就要求换目标。
- **LLM 接口**：`chat(system_text, user_text) -> dict`（JSON）。生产用 `ocres.llm.chat_json`，离线测试可注入假函数。
- Bob 的猜测合法集 = **Alice 可执行集合**（`_alice_pool()`），猜不在集合内记为 None（不计分）。

### 2.3 其它/历史模块（不要混用进主结论）
- `src/ocres/agents.py`：手写规则厨师 `CookAgent`（组 A 用；也被早期脚手架版 LLM 实验 `llm_agent.py` 当执行骨架）。
- `src/ocres/llm_agent.py`：早期"LLM 决策 + 规则 FSM 执行"的脚手架版（`LLMCook` / `LLMBob`），有固定角色 cook/serve。用户后来要求"舍弃状态机、LLM 全权决策"，所以新实验用 `llm_player.py`；`llm_agent.py` 只作对照历史。
- `src/ocres/bob.py`、`data.py`、`cards.py`：组 A 的统计 Bob、数据落盘/loader、早期卡片文本。
- `src/ocres/llm.py`：SiliconFlow 客户端（重试 3 次、超时 300s、JSON 解析容错）。

### 2.4 脚本清单（按时间/用途）
- 环境/布局：`m0_env_check.py`、`m1_probe.py`、`m1_mechanics.py`（机制确认：需显式点火、取汤要先拿盘等）、`m2_geo.py`、`m2_layout_probe.py`、`m2_custom_geo.py`
- 组 A（已被放弃为主结论）：`m2_generate.py`、`m2_twopot.py`、`m3_analyze.py`、`m4_bob_experiment.py`、`m6_*.py`
- 早期 LLM 脚手架：`m5_llm_harness.py`、`m5_live_pilot.py`、`m5_matrix.py`、`m8_*.py`
- **当前主线**：`m7_collect_naive.py`（无经验 Alice 采集，产出 `data/v2_naive/ep*.npz`）、`m7_mine_E.py`（从轨迹提炼经验库 E → `data/v2_E.json`）、`m9_formal.py`（before / after×unaware / after×aware 正式矩阵）
- 调试：`v2_baseline.py`、`v2_debug.py`（打印 trace 与首几条 prompt 到 `data/v2_prompts.txt`）

## 3. 当前实验设计（LLM 主线）

- 布局：自建双锅厨房（两锅、双侧洋葱台、单盘、单出餐口），ASCII 见脚本里的 `ROWS`。
- 指标：
  1. **Bob 猜意准确率**（主指标）：只在 Alice 的决策点计分（Bob 的猜测对齐 Alice 发布的新决定），进一步看"Alice 用了经验(cause)的时刻"这一子集；
  2. 任务得分：每次交付 +20（Overcooked 官方稀疏奖励口径）；
  3. Alice 执行步数（每组实际跑的步数 = 局数 × horizon）。
- 流程：
  1. 无经验 Alice × Bob 跑若干局（`m7_collect_naive.py`）→ 轨迹落盘；
  2. `m7_mine_E.py` 按"结局条件化"从轨迹挖经验条目（现状：E1 空等时去补料、E3 ready 尽快取汤交付；每条带证据 tick 数）；
  3. 正式矩阵（`m9_formal.py`）：before（无 E）/ after_unaware（注入 E，Bob 不知情）/ after_aware（注入 E，Bob 知情并模拟）。

## 4. 目前进展与结果

### 4.1 已稳定跑通
- 环境与全部机制确认；电机层稳定（寻路/朝向/交互/防死锁/容量约束）。
- 部分观测（3×3 + 时间戳锅记忆）与"意图板"协作已上线。
- 无经验 LLM 组现在能出汤（早期全 0 碗，修复容量约束与"持洋葱挡路"后：常见 1–5 碗/300 tick）。
- 正式矩阵（温度 0.1、经验 Top-2 检索注入）修复前最近一轮 n=3：

| 条件 | Bob 全体准确率 | Alice 用经验的时刻 |
|---|---|---|
| before（无 E） | 69.6% | — |
| after_unaware | 79.3% | 75.1%（502 次） |
| after_aware | 86.9% | **86.5%**（637 次） |

方向初步符合预期，但这些数值使用了修复前的重复计数口径，且**样本太小、吞吐波动大**，不能当最终统计。应先按 `decision_id` 事件口径重跑，再讨论趋势。

### 4.2 明确踩过的坑（别再踩）
- `overcooked-ai` 需要 `numpy<2`；不装 `scipy` 会 import 失败。
- 机制：放满 3 个洋葱不会自动开煮，要**空手再 interact 一次**；汤做好后**必须先拿盘子**才能取；交付要在出餐口 interact。
- LLM 长 `reason` 会撑爆 `max_tokens` 导致 JSON 截断 → 现在 `max_tokens≈1600` 且要求 reason ≤30 字。
- 早期 Bob 用"自己的可执行集合"猜 Alice（永远猜不出 FETCH）→ 改为 Alice 合法集后才有意义。
- 早期 Bob 持盘等待时停在 `(2,2)` 咽喉，和 Alice 抢路导致偶发 0 碗 → 停靠点已移开。
- 全 0 碗的另一个根因：两人各揣洋葱、锅满后没人能转身去取盘 → 容量约束 + 持洋葱离位等待解决。
- 硅基流动偶尔 120s+ 超时 → 客户端已重试；跑批请留足时间。

## 5. 需要优化的地方（按优先级）

1. **日志持久化 + 经验质量字段**（最紧要）：目前 `m9_formal.py` 只存汇总，没法评估每条经验好不好。需要：
   - 每局保存逐 tick npz（含 `cause`、Bob `guess`、动作、锅态、交付时刻）；
   - 给经验库每条加统计字段：被采纳次数、采纳后 60 tick 内是否出餐/是否推进、与未采纳对照；
   - 低质量经验自动降权/改写（现在 E1 证据只有 12 tick，偏弱）。
2. **样本量**：每条件 8–10 局，才能谈统计（当前 3 局）。跑一局 2–10 分钟，按允许的时间慢慢堆。
3. **降低随机性**：已做（温度 0.1、结构化 JSON、决策点触发、反自旋）。可再考虑：固定 few-shot 示例、把"可执行集合"写得更明确、必要时对关键决策做多次采样取多数。
4. **去噪**：`m7_mine_E.py` 的检测器阈值与句式可加强；经验文本要"情境—动作—理由—证据"四段齐全，且**只在情境命中时注入**（已做 Top-k 检索，检索打分目前是词面匹配，可换更靠谱的相似度）。
5. **吞吐**：目前"注入经验后吞吐不一定变好"。要么把经验条目写成"效率型"（补锅/时序），要么把吞吐明确降级为参考指标，不用它下结论。
6. **Bob 的强版模拟**：现在只是提示词里让它"模拟"，可以做成两步：先让它写出"我认为 Alice 此刻在做 X，因为她有经验 E#k"，再给猜测，便于审计。
7. **文档/复现**：主结论别引用组 A；`docs/experiments/experiment_design_v2.md`/`docs/experiments/motivation_evidence.md` 里的旧口径需要按"只用 LLM 组"重写一遍。

## 6. 快速上手建议

1. 先跑一次 `PYTHONPATH=src .venv/Scripts/python.exe scripts/m9_formal.py 1` 看单局是否正常出碗、Bob 猜测是否只在 Alice 决策点出现；
2. 看 `data/v2_prompts.txt`（由 `v2_debug.py` 生成）检查提示词是否把状态/方位/可执行集合说清楚；
3. 按优先 1 改造成"逐局落盘 + 经验质量统计"，然后才是加样本量。

有任何不确定，先读 `docs/experiments/experiment_design_v2.md`（设计）与 `docs/experiments/plan_motivation_validation.md`（整体验证计划）。
