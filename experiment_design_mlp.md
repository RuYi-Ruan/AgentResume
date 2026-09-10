# 可训练策略与结构化印象实验设计

## 1. RECOLLAB 给本项目的直接启发

RECOLLAB 将问题拆成三个独立组件：

1. 用 PPO 和不同 reward shaping 训练五种固定队友策略；
2. 在 episode 前 `P=20` 步提取动作直方图、位置停留、首次交互时间、累计奖励等行为指纹，预测离散队友类型；
3. 根据预测类型，从预先训练的 best-response policy library 中只路由一次策略。

因此，论文没有让策略 MLP 读取自然语言“印象”。印象对应的是对离散队友类型的预测或 belief；策略网络只接收环境观测，类型分类器负责选择对应策略。论文还显示：

- 单独使用 rubric 的 COLLAB 不稳定；加入相似轨迹检索后明显改善；
- 逻辑回归在工程化行为指纹上是很强的基线，不能省略；
- 类型识别准确率和最终回报并不完全一致；
- probe 太短信息不足，太长又浪费协作时间；`P=20` 在其实验中最佳；
- `k=3–5` 个检索样例通常足够。

## 2. 与 AgentResume 命题的差别

RECOLLAB 研究“本局开始时遇到哪一种固定的未知队友”；AgentResume 研究“熟悉的伙伴能力已经变化，但观察者仍持有旧印象”。后者必须显式区分：

- Alice 的真实策略或能力 `c_true`；
- Bob 对 Alice 的印象 `b_B(c)`；
- Alice 当前真实意图 `i_t`；
- Bob 基于印象作出的意图预测及协作动作。

不能把“能力变了”和“Bob 猜到类型了”合并成同一个变量。

## 3. 推荐架构

### 3.1 Alice：可训练的分层策略

Alice 的高层策略输出项目已有的八种宏观意图，现有 `Executor` 继续负责原子动作：

```text
observation -> MLP/PPO -> macro intent -> Executor -> primitive action
```

这样既能训练，又保留可审计的意图真值。能力提升前后使用同一网络结构在训练过程中的两个 checkpoint：

- `π_A^pre`：达到基本出汤门槛的早期 checkpoint；
- `π_A^post`：继续训练后，在未见 seed 上稳定提高交付量/平均间隔的 checkpoint。

checkpoint 的选择只由预注册的任务指标决定，不能按 Bob 猜意结果反向挑选。

### 3.2 Bob：印象是 belief，不是文本

第一版把印象表示成离散 one-hot 或概率分布：

```text
z_impression = [P(pre), P(post)]
```

Bob 的意图模型为：

```text
q(intent | observable context, z_impression)
```

如果需要从轨迹更新印象，再增加类似 RECOLLAB 的 probe classifier：

```text
Alice 前 P 步行为 -> fingerprint -> classifier -> z_impression
```

分类器至少比较逻辑回归、MLP、最近邻检索；LLM/RAG 是可选对照，不应是唯一方法。

### 3.3 严格的反事实评估

在同一条 `π_A^post` 测试轨迹上固定环境状态、Alice 行为和真实意图，只改变 Bob 的印象：

- `stale`：强制输入 pre belief；
- `updated`：输入 post belief；
- `inferred`：由前 P 步行为指纹估计 belief；
- `oracle`：直接给真实 post 类型，作为上界；
- `state-only`：不给 impression，检验印象变量是否真的提供额外信息。

主指标是在 Alice 决策事件上的意图准确率，尤其关注“相同可观察局面、pre/post 意图不同”的 reinterpretation set。在线团队回报作为第二阶段指标，避免 Bob 行动改变后续轨迹造成因果混淆。

## 4. 分阶段实验

### Phase A：表示机制验证（已启动）

使用历史 L0/Lk 轨迹，训练纯 NumPy 条件 MLP。输入为 Alice 位置、持有物、两口锅状态和 impression one-hot，输出 Alice 宏观意图。按 seed 留一测试，并在同一 Lk 事件上切换 stale/updated 输入。

这个阶段只回答“MLP 能否使用结构化印象产生可控预测差异”，不证明 Alice 的能力来自学习。

### Phase B：训练 Alice

1. 用历史状态机轨迹行为克隆，得到能稳定完成任务的初始化策略；
2. 用 PPO/IPPO 在稀疏交付奖励和少量合法 shaping 下继续训练；
3. 保存连续 checkpoint，并在独立 seed 上选定 pre/post；
4. 要求 post 在交付量、完成时间或等待率上稳定优于 pre；
5. 收集两个 checkpoint 的新轨迹，重做 Phase A。

### Phase C：印象更新与协作收益

提取与 RECOLLAB 对齐的 probe 指纹，训练类型分类器并进行 `P∈{5,10,20,40,80}` 消融。根据 belief 路由 Bob 的 best-response 策略，比较 stale、updated、inferred、oracle 的团队回报。

## 5. 有效性要求

- 数据按 episode/seed 划分，禁止随机拆 tick；
- 统计单位是决策事件，不是持续多个 tick 的意图状态；
- pre/post 使用相同架构、观测和执行器；
- 训练和评估 seed 分离；
- 报告均值、标准差、置信区间和逐 seed 结果；
- 同时报告分类准确率与团队回报；
- 历史确定性数据只能作为 Phase A 或模仿学习数据，不能作为“学习获得能力提升”的最终证据。
