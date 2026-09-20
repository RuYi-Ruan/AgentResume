# Agent Resume：研究摘要与关键研究点

## 一、研究摘要

在长期多智能体协作中，智能体并非静态不变，而可能随着任务经历不断学习、能力提升并改变其行为模式。已有多智能体记忆方法主要通过保存和检索历史任务、协作轨迹与高层经验来促进系统自身的持续改进。例如，G-Memory 将历史协作组织为 interaction、query 和 insight 三层记忆，并随着新任务持续更新集体经验；DECENTMEM 则为每个智能体维护私有的历史经验与探索记忆，以支持其自身策略演化。另一方面，已有 Ad Hoc Teamwork 工作也注意到队友具有非平稳性，例如 PPAS 通过在线预测持续调整对当前 team model 的判断，从而适应行为发生变化的队友。

然而，这些工作主要关注**如何利用历史经验改进自身决策，或如何快速适应变化后的队友**，较少研究一个更基础的长期协作问题：

> **当同一个队友自身不断成长时，其他智能体过去形成的对其能力和行为模式的认知是否会逐渐过时，以及这种认知滞后会如何影响对其当前行为意图的解释。**

为此，我们研究一种 **Agent Resume** 机制，将其定义为智能体 \(i\) 对特定队友 \(j\) 持续维护的、具有方向性的长期表征：

\[
R_{i\rightarrow j}^{t}
\]

Resume 并非简单存储历史协作轨迹，而是从长期交互证据中逐渐形成并更新关于队友**能力、行为模式、经验与可靠性**的结构化认知。当新的交互证据与已有 Resume 持续不一致时，系统修正对该队友的长期表征，使其能够追踪队友自身的演变。

在决策过程中，智能体进一步将当前观察与最新 Resume 结合，用于解释队友当前动作背后的潜在意图，并据此选择更合适的互补行为：

\[
Observation_t + Action_j^t + R_{i\rightarrow j}^{t}
\rightarrow
Intent_j^t
\rightarrow
Collaborative\ Decision
\]

我们的初步 Overcooked 动机实验已经观察到这一问题：当 Alice 经过训练获得能力提升后，在当前公开状态和第一步动作严格相同的歧义局面中，Bob 使用成长前形成的旧印象时，意图预测准确率仅为 **17.5%**，而使用与成长后 Alice 匹配的新印象时提高到 **70.0%**；不提供历史印象时为 **61.25%**。与此同时，新印象相较旧印象也带来了更高的后续团队得分。

这些结果表明：

> **长期历史并不天然有益。当对队友的认知无法及时跟随队友自身变化时，过时的长期认知甚至可能主动误导当前行为解释。**

因此，本工作关注的并非一般意义上的“记住过去”，而是：

> **如何持续理解一个正在变化的队友，并利用这种长期相互认知改善后续协作。**

---

## 二、核心研究定位

已有工作更多关注：

\[
Past\ Collaboration
\rightarrow
Memory
\rightarrow
Better\ Future\ Action
\]

Agent Resume 关注：

\[
Teammate\ Experience
\rightarrow
Persistent\ Teammate\ Representation
\rightarrow
Intent\ Understanding
\rightarrow
Coordination
\]

核心区别可以概括为：

> **Existing work helps agents remember how they collaborated in the past or adapt their policies to non-stationary teammates. We instead study how agents maintain and revise a persistent understanding of who their teammates are, and how this evolving understanding shapes the interpretation of teammates' current behavior.**

中文：

> **已有工作更多关注“如何记住过去的协作”以及“如何适应变化的队友”，而我们关注的是“如何持续认识一个正在变化的队友”，以及这种长期认知如何帮助智能体理解队友当前行为背后的意图。**

---

## 三、关键研究点

### 1. 从 Task Memory 转向 Teammate Representation

G-Memory、DECENTMEM 等方法的主要研究对象是：

> 过去任务中发生了什么、什么策略成功过、哪些协作轨迹值得复用。

即：

\[
History
\rightarrow
Experience\ Memory
\rightarrow
Better\ Future\ Action
\]

Agent Resume 的研究对象则是**另一个具体智能体本身**：

\[
Interaction\ History_j
\rightarrow
R_{i\rightarrow j}^{t}
\]

即：

> **经过长期合作，我现在认为这个队友是一个怎样的 Agent？**

因此：

- 传统 MAS Memory：**memory for solving tasks**
- Agent Resume：**memory / belief about a teammate**

Resume 需要形成对特定队友的长期表征，例如：

- Capability：这个队友能做什么、擅长什么；
- Behavioral Pattern：这个队友通常如何行动；
- Experience：这个队友经历过哪些类型的任务；
- Reliability：对其某类行为或能力判断有多可信。

---

### 2. 核心现象不是一般 Non-Stationarity，而是 Teammate Representation Staleness

已有 Ad Hoc Teamwork 工作已经研究过：

> teammate 会变化，因此智能体需要适应 non-stationary teammates。

因此，“队友会变化”本身不能作为主要创新点。

我们更进一步关注：

\[
Z_j^t \neq R_{i\rightarrow j}^{t-\Delta}
\]

其中：

- \(Z_j^t\)：队友 \(j\) 当前真实的能力与行为状态；
- \(R_{i\rightarrow j}^{t-\Delta}\)：Agent \(i\) 仍然持有的旧认知。

我们将这种现象定义为：

## **Teammate Representation Staleness**

即：

> **队友已经发生变化，但其他智能体对他的长期认知仍然停留在过去。**

我们关心的不只是最终 reward 是否下降，而是这种认知滞后如何造成：

\[
Stale\ Resume
\rightarrow
Intent\ Misinterpretation
\rightarrow
Wrong\ Collaborative\ Decision
\rightarrow
Coordination\ Degradation
\]

当前 Overcooked 动机实验已经观察到：

\[
Stale\ Resume < No\ Resume
\]

这说明：

> **错误的长期认知可能比完全没有长期认知更有害。**

---

### 3. Resume 的核心任务是“修正对队友的认识”，而不是简单追加历史

普通 memory update 通常是：

\[
M_t + New\ Trajectory
\rightarrow
M_{t+1}
\]

但 Agent Resume 不能只是把新轨迹继续加入 memory。

真正要研究的是：

\[
Evidence_{1:t}
\rightarrow
Change\ Detection
\rightarrow
Resume\ Revision
\]

也就是说，需要回答：

- 什么时候应该认为队友已经发生变化？
- 哪些 Resume 字段需要更新？
- 一次新证据应该改变多少旧认知？
- 如何区分偶然行为与稳定能力变化？
- 如何避免被少量异常轨迹误导？
- 如何表示对当前 Resume 的置信度？
- 一个旧认知何时应该被降权、替代或废弃？

因此 RQ1 的重点不是：

> 如何存储更多历史？

而是：

> **如何根据持续到来的任务经历，稳定地修正对一个 evolving teammate 的长期认知？**

---

### 4. 从“适应队友”进一步研究“理解队友”

PPAS 等 non-stationary teammate 方法更接近：

\[
Teammate\ Changed
\rightarrow
Identify/Reweight\ Team\ Models
\rightarrow
Adapt\ Own\ Policy
\]

其核心问题是：

> **队友变化以后，我该怎么做？**

Agent Resume 显式增加一个中间层：

\[
Observation_t
+
Action_j^t
+
R_{i\rightarrow j}^{t}
\rightarrow
Intent_j^t
\rightarrow
Collaborative\ Decision
\]

我们研究的是：

> **基于我长期以来对这个具体队友的认识，他当前这个动作究竟意味着什么？**

因此需要区分：

### Adaptation
> What should I do given that you changed?

### Understanding
> Given who I believe you are now, what does your current behavior mean?

Agent Resume 希望实现：

\[
Understanding
\rightarrow
Adaptation
\]

而不是直接从观察跳到动作适应。

---

### 5. Capability、Behavior 与 Intent 必须区分

Agent Resume 中至少需要区分三个层次：

#### Capability
> **Can the teammate do it?**

例如：

- Fetch Skill = 0.92
- Serving Skill = 0.74

#### Behavioral Pattern
> **What does the teammate usually do?**

例如：

- 在锅烹饪期间更偏好继续取料；
- 通常完成当前子任务后再切换；
- 很少中途打断 teammate。

#### Intent
> **What is the teammate trying to do now?**

Intent 不直接存储为长期静态属性，而应由：

\[
Current\ Observation
+
Current\ Action
+
Resume
\]

进行推断。

即：

\[
Capability + Behavior
\rightarrow
Prior\ Understanding
\]

\[
Prior\ Understanding + Current\ Evidence
\rightarrow
Intent
\]

这样可以避免把：

- “能不能做”
- “平时怎么做”
- “这一次想做什么”

混为一谈。

---

## 四、建议的研究问题

### RQ1：Resume Evolution

> **当队友的能力和行为模式随着任务经历持续变化时，Agent Resume 如何及时、准确且稳定地更新对该队友的长期表征？**

重点关注：

- Change Detection
- Update Speed
- Adaptation Lag
- Profile Accuracy
- Confidence Calibration
- Resistance to Noisy Evidence

可以定义：

\[
Adaptation\ Lag
=
t_{resume\ updated}
-
t_{teammate\ changed}
\]

即：

> 队友已经发生变化之后，需要经过多少次新的交互，Resume 才能准确跟上？

---

### RQ2：Resume-grounded Intent Understanding

> **与没有 Resume 或使用过时 Resume 相比，持续更新的 Resume 是否能够帮助智能体更准确地解释队友当前行为背后的意图？**

形式化为：

\[
P(Intent_j^t
\mid
Observation_t, Action_j^t, R_{i\rightarrow j}^{t})
\]

与：

\[
P(Intent_j^t
\mid
Observation_t, Action_j^t)
\]

比较。

核心指标包括：

- Intent Accuracy
- Target Accuracy
- Calibration
- Ambiguous-state Intent Accuracy

---

### RQ3：Understanding-to-Coordination

> **更准确的队友意图理解是否能够进一步转化为更高质量的协作决策与团队表现？**

即验证：

\[
Resume\ Accuracy
\rightarrow
Intent\ Accuracy
\rightarrow
Coordination\ Quality
\]

可评估：

- Team Reward
- Task Success Rate
- Duplicate Action Rate
- Conflict Rate
- Complementary Action Rate
- Completion Time
- Coordination Efficiency

---

## 五、与代表性已有工作的区别

### 1. 与 G-Memory 的区别

G-Memory 主要解决：

> 如何组织、检索并持续更新长期 MAS 协作历史，使未来任务能够利用过去的经验。

其核心是：

\[
Interaction\ Graph
+
Query\ Graph
+
Insight\ Graph
\]

并面向不同角色提供定制化 memory。

Agent Resume 与其不同：

\[
G\text{-}Memory:
Memory\ tailored\ FOR\ Agent_i
\]

\[
Agent\ Resume:
Representation\ ABOUT\ Agent_j
\]

前者是：

> 给某个 Agent 什么历史经验最有帮助？

后者是：

> 我现在认为某个具体 teammate 是怎样的？

因此 Agent Resume 建模的是：

\[
R_{i\rightarrow j}
\]

这种**具有方向性的 teammate representation**。

---

### 2. 与 DECENTMEM 的区别

DECENTMEM 为每个 Agent 保存自己的：

- exploitation memory；
- exploration memory；
- historical collaboration trajectory。

目标是让每个 Agent：

\[
Own\ Experience
\rightarrow
Own\ Memory
\rightarrow
Better\ Own\ Strategy
\]

Agent Resume 则是：

\[
Teammate_j's\ Experience
\rightarrow
Belief_{i\rightarrow j}
\]

DECENTMEM 更偏：

> **Self Memory**

Agent Resume 更偏：

> **Social / Teammate Memory**

此外，DECENTMEM 的在线更新主要调整 exploitation 与 exploration 的权重，而不是持续修正对某一个 teammate 的能力、行为与可靠性判断。

---

### 3. 与 Governed Shared Memory 的区别

Governed Shared Memory 研究的是：

> 多智能体共享 memory 中事实如何保持 temporal correctness、consistency、provenance 和 access control。

例如：

\[
Address=v_1
\rightarrow
Address=v_2
\]

其他 Agent 是否仍然读到：

\[
v_1
\]

这属于：

## **Distributed-state staleness**

Agent Resume 的 stale 则是：

> Alice 已经改变，但 Bob 并没有被显式通知，只能通过行为证据推断 Alice 是否已经变化。

因此属于：

## **Epistemic / Cognitive Staleness**

即：

\[
Observation
\rightarrow
Evidence
\rightarrow
Infer\ Change
\rightarrow
Update\ Belief
\]

两者关注层次不同。

---

### 4. 与 PPAS / Non-Stationary Teammate Adaptation 的区别

PPAS 已经研究：

> 如何在 teammate policy 非平稳时，根据新观察在线调整对 team model 的判断，并改变自己的策略。

其核心更接近：

\[
Observation
\rightarrow
Team\ Model\ Reweighting
\rightarrow
Policy\ Adaptation
\]

Agent Resume 则进一步关注：

\[
Teammate\ Experience
\rightarrow
Persistent\ Individual\ Representation
\rightarrow
Intent\ Attribution
\rightarrow
Coordination
\]

PPAS 主要回答：

> **Which model best explains this teammate now?**

Agent Resume 更进一步回答：

> **Who has this teammate become, and what does their current behavior mean given that evolving identity?**

因此我们不是简单重复：

> non-stationary teammate adaptation

而是研究：

> **longitudinal teammate understanding**

---

## 六、当前 Overcooked 动机实验所支持的关键结论

当前实验构造了严格配对的歧义局面：

- 成长前和成长后 Alice 面对相同厨房状态；
- 两者位置、朝向、手持物相同；
- Bob 的局部观察相同；
- Alice 的第一步动作相同；
- 第一步后的公开状态相同；
- 但成长前 Alice 的真实意图为 `PRE`；
- 成长后 Alice 的真实意图为 `FETCH`。

因此：

> Bob 不能只依赖当前一步动作判断答案，必须依赖过去对 Alice 形成的认知。

正式结果：

| Bob 条件 | 意图准确率 | 后续平均总分 |
|---|---:|---:|
| 旧印象 | 17.50% | 100.5 |
| 新印象 | 70.00% | 106.0 |
| 无印象 | 61.25% | 104.0 |

这支持：

\[
Stale\ Representation
\rightarrow
Intent\ Misinterpretation
\rightarrow
Coordination\ Degradation
\]

特别是：

\[
No\ Resume > Stale\ Resume
\]

说明：

> **过时的 teammate representation 不只是“没有帮助”，而可能主动误导协作。**

因此当前实验非常适合作为论文的：

## **Motivating Experiment**

用于回答：

> **Why does an Agent Resume need to evolve?**

---

## 七、最终研究主线

整个工作可以统一为：

\[
\boxed{
Teammate\ Experience
\rightarrow
Evolving\ Resume
\rightarrow
Current\ Intent\ Attribution
\rightarrow
Personalized\ Coordination
}
\]

进一步展开：

\[
Teammate\ Evolves
\]

\[
\downarrow
\]

\[
Old\ Resume\ Becomes\ Stale
\]

\[
\downarrow
\]

\[
Resume\ Detects\ Behavioral/Capability\ Change
\]

\[
\downarrow
\]

\[
Resume\ Is\ Revised
\]

\[
\downarrow
\]

\[
Better\ Interpretation\ of\ Current\ Behavior
\]

\[
\downarrow
\]

\[
Better\ Intent\ Attribution
\]

\[
\downarrow
\]

\[
Better\ Coordination
\]

---

## 八、论文核心定位一句话

### 英文

> **Existing work helps agents remember how they collaborated in the past or adapt their policies to non-stationary teammates. We instead study how agents maintain and revise a persistent understanding of who their teammates are, and how this evolving understanding shapes the interpretation of teammates' current behavior.**

### 中文

> **已有工作更多关注如何记住过去的协作，或如何适应变化后的队友；我们关注的是如何持续认识一个正在变化的队友，以及这种不断演化的长期认知如何帮助智能体解释队友当前行为背后的意图。**
