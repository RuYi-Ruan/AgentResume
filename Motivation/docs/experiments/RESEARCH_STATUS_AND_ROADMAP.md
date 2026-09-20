# AgentResume 研究现状与下一阶段路线

更新日期：2026-09-10

本文记录截至当前的研究讨论、已经获得的证据、尚未解决的问题，以及环境就绪后的下一阶段实验计划。它是研究决策记录；具体代码结构见 `PROJECT_HANDOFF.md`，旧实验细节见其他设计文档。

## 1. 当前要验证的核心命题

伙伴 Alice 的能力通过训练得到提升后，其在某些相同或相近局面下的行为与意图会发生变化。如果 Bob 对 Alice 的印象仍停留在提升前，Bob 会继续使用旧的“情境—行为—意图”映射，因此对 Alice 当前行为背后意图的推测会系统性失准；当 Bob 更新了对 Alice 的印象后，推测准确率应恢复。

这个命题包含三个必须分开控制的变量：

- Alice 的真实能力或策略：`c_true ∈ {pre, post}`；
- Bob 对 Alice 的印象（belief）：`b_B(c)`；
- Alice 在当前决策事件上的真实宏观意图：`i_t`。

因此，不能再把“给 Alice 注入了一段经验文本”直接等同于“Alice 的能力已经稳定提高”，也不能把 Bob 的印象与 Bob 自己的控制策略混成一个变量。

## 2. 研究路线如何演进

### 2.1 确定性状态机阶段

早期 `CookAgent` 用显式规则构造了串行烹饪的 L0 Alice 和并行烹饪的 Lk Alice。这条管线稳定、可解释，也观察到了旧印象在重解释点上失败的现象。

它的价值是证明机制可能存在，并提供训练数据、调试工具和实验设计原型；局限是能力提升由人工改规则实现，不是智能体通过训练真正获得的。因此，它不能作为最终论文中“学习导致能力提升”的主要证据。

### 2.2 LLM 阶段

当前 `LLMPlayer` 通过给 Alice 注入自然语言经验 E 来改变决策，Bob 则在 unaware/aware 条件下猜测 Alice 的宏观意图。LLM 的优势是能够直接读取文字经验和文字印象，研究语义直观。

实验同时暴露了核心问题：经验注入能够改变行为，但任务能力提升不稳定，吞吐量受采样、提示词、模型服务和局部死锁影响较大。因此，LLM 很难单独承担严格的 `pre → post` 能力操纵。LLM 路线仍保留为语言型对照和后续扩展，但暂不作为下一阶段的主要因果验证载体。

### 2.3 RECOLLAB 与可训练策略阶段

RECOLLAB 的关键启发不是“让 MLP 理解文字印象”，而是把印象表示为对伙伴策略类型的 belief：先由行为轨迹识别伙伴类型，再选择对应的响应策略。

对 AgentResume 更合适的做法是：

```text
Alice: structured observation -> trainable policy -> macro intent -> Executor
Bob:   Alice history -> belief over {pre, post} -> intent predictor / response policy
```

MLP 不需要理解“她以前很弱”这样的自然语言。第一版直接输入结构化印象，例如 `[P(pre), P(post)]`；以后若要支持文字，可使用冻结文本编码器先把文字转为 embedding，再交给 MLP。

## 3. 当前证据应如何解释

### 3.1 仍然有效的工程与方法基础

- Overcooked 自定义双锅环境、部分观测、寻路和原子执行层已经可运行；
- Alice 的决策事件使用唯一 `decision_id`，Bob 的一次性 `guess_event` 按 ID 精确配对；
- 指标已从“逐 tick 重复状态”修复为“每个真实决策事件计一次”；
- 纯 NumPy 条件 MLP 已证明结构化 impression 可以控制意图预测；
- 测试按 episode/seed 划分，而不是随机拆分 tick。

### 3.2 只能作为机制或可行性证据的结果

历史状态机轨迹上的条件 MLP 留一 seed 结果为：

| 条件（固定在 Lk 测试事件上） | 全体意图准确率 | 重解释子集准确率 |
|---|---:|---:|
| stale L0 impression | 0.821 | 0.000 |
| updated Lk impression | 1.000 | 1.000 |
| state-only | 0.937 | 0.000 |

这说明模型确实会使用 impression，而且同一事件仅切换 impression 就能改变预测；但数据来自人工脚本策略且轨迹高度同构，不能据此宣称“训练后的伙伴能力提升导致印象滞后”。

LLM 旧结果只说明 aware 通常比 unaware 更容易理解注入经验后的 Alice。由于样本小、策略随机且旧批次曾有重复计数问题，`data/v2_results_formal.json` 不能作为正式统计。

## 4. 精化后的可检验假设

- **H1 能力提升**：在与训练 seed 分离的评估集上，`π_A^post` 的交付量、完成时间或等待率稳定优于 `π_A^pre`。
- **H2 行为—意图映射漂移**：pre/post 策略存在足够数量的重解释事件，即在匹配的可观察上下文中，真实宏观意图不同。
- **H3 印象滞后效应**：在完全相同的 post 轨迹上，输入 stale/pre belief 的 Bob 意图准确率低于输入 updated/post belief 的 Bob。
- **H4 更新恢复**：updated 或从新轨迹 inferred 的 belief 能显著缩小 stale 与 oracle 之间的差距。
- **H5 协作后果（第二阶段）**：当 Bob 根据错误印象采取行动时，团队回报或协作效率下降。

H3 是当前 motivation 的首要指标；H1 和 H2 是解释 H3 的必要前提。H5 重要，但应在固定轨迹的离线反事实验证完成后再做，避免 Bob 的行为反过来改变 Alice 所处状态。

## 5. 下一阶段实验：真正训练出 pre/post Alice

### 5.1 Alice 的策略形式

第一版使用小型 PyTorch MLP 输出现有宏观意图，继续复用确定性的 `Executor`：

```text
structured observation
    -> MLP policy + legal-action mask
    -> FETCH / PLACE / COOK_START / GET_DISH / PICKUP / DELIVER / PRE / HOLD
    -> existing Executor
```

输入使用结构化数值特征，而非画面。M11 历史数据基线使用日志可无歧义重建的紧凑全局状态（双方位置/手持、两锅状态、时间和交付数）；首轮因果验证也优先采用该观测，以排除感知误差这个额外混淆。当前 LLM 的 3×3 部分观测将在主效应成立后作为鲁棒性扩展，而不会与全局状态结果混报。网络规模很小，MX450 的 2 GB 显存足以进行行为克隆和小规模 PPO；必要时环境 rollout 在 CPU、网络更新在 GPU。

### 5.2 训练顺序

1. 从历史可行轨迹建立统一的 observation/action 数据接口；
2. 行为克隆得到能基本完成任务的初始化策略，验证网络、标签和 action mask；
3. 在同一网络上继续 PPO/IPPO 训练，定期保存 checkpoint；
4. 仅按预注册的任务能力指标，从训练曲线中选择 `pre` 和 `post`；
5. 用未见 seed 对两个 checkpoint 进行独立评估并采集决策事件；
6. 只有 H1 达标后，才训练和评估 Bob 的印象条件意图模型。

行为克隆阶段是工程与表示验证。最终的 pre/post 应来自同一次可训练策略的学习过程，而不是分别模仿人工定义的 L0/Lk 后把两者称为能力成长。

### 5.3 Bob 的印象与严格反事实

初版 Bob 意图模型为：

```text
q(i_t | observable_context_t, Alice_history, z_impression)
z_impression = [P(pre), P(post)]
```

在同一条 `π_A^post` 测试轨迹上冻结状态、Alice 行为、历史和真实意图，只切换：

- `stale`：`z=[1,0]`；
- `updated`：`z=[0,1]`；
- `state-only`：移除或置零 impression；
- `oracle`：直接提供真实 post 类型；
- `inferred`：后续由前 P 步行为指纹预测 belief。

这样 stale/updated 的差异不能归因于不同 rollout，是最直接的印象滞后因果检验。

## 6. 数据划分、指标与验收门槛

必须遵守：

- 训练、验证、测试按 episode/seed 分离；
- 统计单位为 Alice 的唯一决策事件；
- pre/post 使用相同网络结构、观测接口、执行器和评估 seed；
- checkpoint 不能根据 Bob 的猜意结果挑选；
- 同时报告逐 seed 结果、均值、标准差和 bootstrap 置信区间；
- 主结果同时提供全事件集与预先定义的 reinterpretation set。

进入 Bob 正式实验前的建议门槛：

1. post 在至少两个任务能力指标上优于 pre，且多数独立 seed 同向；
2. pre/post 均能稳定完成基本任务，避免把“不会做饭”误当作低能力；
3. 测试集中有足量重解释事件，而非只靠几个偶然样本；
4. Alice 意图标签来自策略实际选择的宏观动作，不用事后结果反推；
5. state-only、逻辑回归等简单基线必须保留。

## 7. 近期实施里程碑

### M11：PyTorch 数据与策略接口

状态：**基础里程碑已完成（2026-09-10）**。

- 固定 observation schema 和 intent vocabulary；
- 实现 action mask、Dataset/DataLoader 和小型 MLP policy；
- 加入 CPU/GPU 自动选择、随机种子和 checkpoint 格式；
- 用少量历史数据过拟合，确认数据—标签—执行链路正确。

完成结果：使用 L0 的 7 个训练 seed（730 个决策事件）训练，在留出的 seed 8（104 个事件）上八类意图准确率均为 100%，训练设备为 MX450/CUDA。checkpoint 写入 `artifacts/checkpoints/m11_bc_l0.pt`，汇总见 `data/m11_bc_results.json`。该结果只验收接口、动作掩码、GPU 和 checkpoint；历史 seed 高度同构，因此不构成策略泛化或能力成长证据。

### M12：行为克隆基线

状态：**已完成闭环基线（2026-09-10）**。

- 按 episode 划分数据；
- 报告宏观意图准确率、混淆矩阵及闭环交付能力；
- 检查模型是否只记住 seed/位置模板。

闭环结果：在 seeds 1–16、每局 700 tick 的同起点对照中，L0 专家和 BC Alice 均为每局 15 碗，零失败局；BC Alice 平均交付间隔为 42.41 tick，专家为 45.0。初版适配器曾在 11/16 局于 2 碗后因双向占位死锁，加入“连续阻塞后让位并保持 3 tick”的纯执行层护栏后，16/16 局恢复稳定。该修复不改变模型宏观意图。

离线 100% 与闭环稳定共同说明 checkpoint 可作为 PPO 的 `pre` 初始化；它仍然来自脚本 L0 行为克隆，而且 BC 在部分分布外状态会选择不同于串行专家的动作，因此不能称为训练后的能力提升。完整汇总见 `data/m12_closed_loop_results.json`。

### M13：PPO 训练与 checkpoint 选择

状态：**H1 已通过（2026-09-11）**。

- 以稀疏交付奖励为主，shaping 只用于解决极端稀疏；
- 保存连续 checkpoint 和完整配置；
- 在固定未见 seed 上形成能力曲线并锁定 pre/post。

实现采用宏观 semi-MDP PPO：一次策略动作对应一个宏观意图，执行器运行到意图完成；GAE 的折扣按实际 primitive tick 持续时间计算。训练只使用团队稀疏交付奖励，validation seeds 用于选择 checkpoint，独立 test seeds 不参与训练或选择，Bob 指标完全未使用。

正式 30-update 结果：pre 在 test seeds 3001–3016 上均为 15 碗/700 tick；post 均值 24.625 碗、范围 23–25。16/16 个 seed 同向提升，配对均值差为 +9.625 碗，episode bootstrap 95% CI 为 `[9.3125, 9.875]`。因此 H1“同一可训练 Alice 的后期 checkpoint 稳定优于早期 checkpoint”通过。完整曲线见 `data/m13_ppo_results.json`。

### M14：印象滞后正式实验

状态：**H2/H3 通过，H4 完成 oracle-updated 部分（2026-09-11）**。

- 用真实训练 checkpoint 重建 conditional intent predictor；
- 完成 stale/updated/state-only/oracle 的固定轨迹反事实；
- 再加入类似 RECOLLAB 的 probe classifier，研究印象需要多少新行为才能更新。

正式数据使用 train seeds 4001–4024 和独立 test seeds 5001–5016。共采集 8,342 个训练决策事件；测试包含 2,475 个 pre 和 3,089 个 post 决策事件。在完全相同的 post 测试事件上只切换 belief：

| 条件 | 全部 post 事件准确率 | 39 个重解释事件准确率 |
|---|---:|---:|
| stale/pre belief | 66.49% | 23.08% |
| updated/post belief | 100% | 100% |
| state-only | 99.61% | 71.79% |

updated 相对 stale 的逐 seed 配对提升在 16/16 个 seed 上同向：全事件均值差 33.51 个百分点，bootstrap 95% CI `[33.24, 33.75]`；重解释子集按 seed 平均提升 73.96 个百分点，95% CI `[61.46, 87.5]`。因此，真实训练 checkpoint 已产生行为—意图映射漂移（H2），强制保留旧 belief 会造成系统性误读（H3）。

解释边界：state-only 在全部日常事件上已接近满分，印象的额外价值集中在重解释点；updated 当前是外部给定的正确 one-hot belief，属于 oracle 更新，不等于 Bob 已能自行发现 Alice 成长。下一步应训练 probe classifier 并做 `P` 长度消融，完成 H4 的 inferred 条件。结果见 `data/m14_checkpoint_impression_results.json`。

## 8. 当前环境状态

主要实验环境已经就绪：

```text
Conda env: C:\Users\86134\.conda\envs\agentresume
Python:    3.11
PyTorch:   2.1.0+cu118
CUDA:      available
GPU:       NVIDIA GeForce MX450, 2 GB
NumPy:     1.26.4
SciPy:     1.17.1
```

2026-09-11 已在禁用用户级 site-packages 的情况下运行全部 12 个离线测试，全部通过。PyTorch wheel 保存在 `D:\torch-wheel`，可用于离线重装。正式 pre/post checkpoint 体积均小于 60 KB，随仓库保存以便直接复核 M14。

Conda 的 NumPy/MKL 与官方 Windows PyTorch wheel 都携带 Intel OpenMP。训练入口在导入数值库前设置 `MKL_THREADING_LAYER=SEQUENTIAL`，避免同一进程加载两份 runtime；这不使用不安全的 `KMP_DUPLICATE_LIB_OK`，也不影响 CUDA kernel。交互式混用 NumPy/PyTorch 时也应在首次导入二者前设置该变量。

推荐运行方式：

```powershell
conda activate agentresume
$env:PYTHONNOUSERSITE = "1"
$env:MKL_THREADING_LAYER = "SEQUENTIAL"
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

## 9. 当前研究决策

1. 主 motivation 不变：研究伙伴成长后的印象滞后与意图误读；
2. 下一主线改为“可训练 Alice + 结构化 Bob belief”，以获得稳定、可复现的能力操纵；
3. LLM 保留为文字经验/印象的扩展对照，不承担唯一的能力提升证据；
4. 状态机数据保留为行为克隆、机制分析和回归测试素材，不作为学习成长的最终证明；
5. 下一项代码工作从 M11 开始，而不是继续扩大旧 LLM 汇总结果。
