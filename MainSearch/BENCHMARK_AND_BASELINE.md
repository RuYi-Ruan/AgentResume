# 主实验 Benchmark 与 Baseline 初步选择

## 当前判断

主实验优先考虑 **Hanabi**，困难版 **Overcooked** 作为第二个环境。

选择标准不是游戏看起来是否复杂，而是：Bob 对 Alice 的判断一旦过时，是否会真实影响两人的合作结果。

## Benchmark 建议

| 环境 | 优点 | 问题 | 建议 |
|---|---|---|---|
| Hanabi | 必须通过伙伴的提示和动作理解其想法，误解会直接造成失误和掉分 | 需要设计可控的成长前、成长后伙伴 | 主实验首选 |
| 困难版 Overcooked | 已有代码和实验基础；能直观看到堵塞、重复劳动和分工失败 | 简单地图上伙伴建模对出餐影响较小 | 第二环境 |
| Melting Pot | 伙伴差异和社会互动丰富 | 环境较重，意图真值不容易确定 | 暂缓 |
| Minecraft | 任务丰富，合作空间大 | 成本高；直接交流可能让意图推测失去必要性 | 暂缓 |
| ALFWorld | LLM Agent 常用，任务和评测比较成熟 | 原本主要是单智能体任务；拆成多个角色不等于真正需要理解伙伴 | 不作为主实验首选 |

Hanabi 虽然提出较早，但问题没有过时。2025 年的 AH2AC2 仍然选择 Hanabi 研究陌生伙伴合作，说明它仍是研究伙伴理解和临时组队的有效环境。主实验可以让同一个 Alice 从早期训练模型切换到后期训练模型，Bob 不会收到通知，只能从后续行为中发现变化并更新印象。

## Baseline 建议

建议第一版至少包含：

1. **No Resume**：完全不使用长期印象。
2. **Frozen Resume**：始终使用成长前形成的旧印象。
3. **Full History**：不整理，直接使用全部历史。
4. **Sliding Window**：只保留最近若干局。
5. **PLASTIC**：根据新行为持续调整对伙伴类型的判断。
6. **RECOLLAB**：检索相似轨迹后判断伙伴类型，是与本项目最接近的直接 baseline。
7. **G-Memory**：代表通用多智能体长期记忆，用来判断“记住协作历史”是否足以代替“维护特定伙伴的印象”。
8. **Oracle**：直接获得 Alice 当前版本，作为理想上限。
9. **Agent Resume（我们的方法）**：检测伙伴变化，修正结构化印象，再用于理解行为和选择配合方式。

## 阅读文件中的方法是否适合

- **G-Memory：适合做通用记忆 baseline。** 它保存任务、经验和协作轨迹，但重点不是持续认识某个正在成长的伙伴。
- **DECENTMEM：可作为补充。** 它强调每个智能体分别保存自己的经验，与 Bob 保存“关于 Alice 的认识”仍有差别。
- **Governed Shared Memory：不适合作为核心算法 baseline。** 它主要处理权限、同步、来源和旧数据失效等系统问题。
- **RECOLLAB：最值得保留。** 它直接根据伙伴轨迹识别伙伴类型并切换配合策略，但没有重点处理同一个伙伴长期成长后旧认知过时的问题。

## AutoGen 与 ALFWorld 的位置

- **AutoGen 是搭建多智能体流程的框架，不是 benchmark。** 它相当于实验系统的“骨架”。
- **ALFWorld 是 benchmark。** 它让 Agent 在文字化的家庭环境中完成拿取、加热、清洁、放置等任务。
- G-Memory、DECENTMEM 经常使用它们，是因为组合成熟、方便比较记忆方法；这不代表它们天然最适合研究“同一个伙伴成长后，旧印象是否过时”。

## 初步路线

先做一个小规模 Hanabi 可行性实验，确认三件事：

1. Alice 的早期和后期模型确实表现出可观察的行为变化；
2. Bob 使用旧印象时，合作得分会明显下降；
3. 更新印象后，Bob 的判断和团队得分都能恢复。

这三点成立后，再正式确定 Hanabi 为主实验，并在困难版 Overcooked 上验证方法不是只对一种游戏有效。

## 参考

- [The Hanabi Challenge](https://arxiv.org/abs/1902.00506)
- [Ad-Hoc Human-AI Coordination Challenge (2025)](https://arxiv.org/abs/2506.21490)
- [Melting Pot 2.0](https://arxiv.org/abs/2211.13746)
- [G-Memory](https://arxiv.org/abs/2506.07398)
- [Overcooked Generalisation Challenge](https://arxiv.org/abs/2406.17949)
