# M16 当前进度与暂停点

## 已完成

### 地图与观察规则

- 最终候选布局：`ambiguous_kitchen_v1`，11×7，设施交错；
- 3×3 视野；
- Bob 看不到 Alice 时不会收到意图变化通知；
- 每段意图最多预测一次：使用该段中 Bob 首次亲眼看到的新动作；
- 看不到整段行为时不猜、不计入准确率。

地图验证结果保存在 `data/m16_layout_validation.json`：

- 几何第一动作歧义率约 78.4%；
- 保守脚本参考 13 碗；
- 并行脚本参考 22 碗；
- 所有设施可达。

### 通用执行修复

- 修复边缘设施前无法原地转向交互的问题；
- 修复两人反复争抢同一格但控制器没有识别实际阻塞的问题；
- 修复后 22 项测试全部通过。

### Alice-pre

- 使用保守专家行为克隆；
- 182 维新地图观测；
- 训练事件 3,335，独立离线测试事件 2,220；
- 训练/测试准确率均为 100%；
- 闭环独立测试 16/16 seed 均为 13 碗。

结果：`data/m16_alice_pre_results.json`；checkpoint：`artifacts/checkpoints/m16_alice_pre.pt`。

### Alice-post

第一次 PPO（40轮、原探索强度）没有超过 pre，按门槛判定失败。

第二次只调整探索强度，不增加中间奖励：

- temperature 5.0；
- entropy coefficient 0.08；
- 使用全新的 validation seeds 8501–8516 和 test seeds 8601–8616；
- 第35轮验证达到17碗，第45轮19碗，第50轮22碗；
- 最终独立测试 pre=13、post=22；
- 16/16 seed 都提高9碗。

结果：`data/m16_alice_post_results.json`；checkpoint：`artifacts/checkpoints/m16_alice_post.pt`。

## 冻结 checkpoint 后发现的问题

在新的、未用于训练或选模的 seeds 8801–8816 上，使用固定上菜伙伴检查 Bob 的真实3×3可见事件：

- pre Alice：平均每局约 21.9 个可见意图片段；可见标签只有 `HOLD/FETCH`；
- post Alice：平均每局约 18.9 个可见意图片段；可见标签只有 `FETCH/PLACE`；
- pre 的 `HOLD` 基本表现为 `(0,0)` 停留，`FETCH` 基本表现为水平移动；
- post 的 `FETCH` 发生时 Alice 空手，`PLACE` 发生时 Alice 手持洋葱。

因此，尽管地图在纯几何上有歧义，真实策略产生的“可见状态 + 动作”仍然几乎直接暴露意图。此时调用 LLM 很可能重复 M15：模型不需要使用 impression。

## 建议修正（待用户审核）

让成长前 Alice 在“一口锅烹饪、另一口锅仍可填料”的关键状态下，不再用原地 `HOLD` 表示保守等待，而是使用 `PRE`：走到当前烹饪锅附近的安全等待区，观察并准备下一步。成长后 Alice 则使用 `FETCH`，去拿洋葱并行填另一口锅。

地图相应保证：前往烹饪锅等待区和前往洋葱台的路线，在开始阶段共用至少一个动作。这样 Bob 看到第一步移动时，不能仅凭“动了还是没动”判断 Alice 是守锅还是准备第二锅，需要结合历史 impression。

约束：

- `PRE` 目标是“当前烹饪锅附近”，不是隐藏心理状态；
- Bob 仍只看到 Alice 的位置、手持物和实际动作；
- Bob 不会看到 `PRE/FETCH` 标签或内部 target；
- 等待区不能占据锅的交互站位或主通道；
- 重新训练 Alice-pre/post，并重新通过能力提升与可见歧义门槛；
- 在证明同一第一动作确实对应 `PRE/FETCH` 两种意图前，不调用 Qwen。

## 当前状态

实验停在 LLM 调用之前。Qwen3.5-9B 和 API 配置已确认存在，但尚未产生新的调用费用。
