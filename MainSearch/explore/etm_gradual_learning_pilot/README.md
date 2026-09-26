# ETM 渐进学习机制小实验

这个目录用于验证一个很窄的问题：同一批伙伴在持续协作中通过在线学习逐渐提升时，动态更新的 Evolving Teammate Model（ETM）是否比冻结认知更有利于意图判断和协作。

这不是正式 benchmark，也不能作为论文结论。它是一个可控的机制实验，用来决定后续是否值得在真实多人环境中实现 ETM。

## 实验设置

- 团队中有 3 个固定身份的智能体：焦点智能体 A，以及伙伴 B、C。
- 每轮有 3 个角色，团队只有在 3 个角色都被覆盖时才获得协作成功。
- B、C 不切换 checkpoint。它们各自维护一张 Q 表，在每轮协作后进行一次小幅在线更新，逐渐学会在不同情境下选择自己的目标角色。
- A 在决策前只能看到 B、C 带噪声的行为线索，需要推断它们本轮选择的角色，再选择互补角色。
- 每轮结束后，伙伴实际选择的角色变为可观察证据，供动态 ETM 更新。

比较四种条件：

1. `no_model`：只相信当前带噪声线索，不使用伙伴历史；
2. `frozen_etm`：只在最初若干轮建立队友模型，之后冻结；
3. `dynamic_etm`：持续按伙伴身份和情境更新队友模型；
4. `oracle`：直接知道伙伴本轮真实意图，作为上界。

四种条件共享完全相同的伙伴选择和观察线索，因此差异只来自 A 如何理解伙伴。

## 运行

在仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe MainSearch\explore\etm_gradual_learning_pilot\run_experiment.py
```

运行后生成：

- `results/summary.json`：总体结果、分阶段结果和配对检验；
- `results/learning_curve.csv`：按时间窗口汇总的伙伴能力、意图准确率与协作成功率。

运行测试：

```powershell
.\.venv\Scripts\python.exe -m unittest MainSearch.explore.etm_gradual_learning_pilot.test_experiment
```

## 这个实验不能说明什么

- 伙伴学习的是简化的情境—角色映射，不是真实 Overcooked 策略；
- A 的行动不会反过来影响 B、C 的学习，因此还没有完整的共同适应；
- 每轮结束后真实角色可观察，真实环境中可能只能得到间接证据；
- ETM 目前只是带遗忘因子的身份条件概率模型，不是最终方法。

只有当动态 ETM 在伙伴能力平滑增长期间稳定优于冻结模型，而且在能力不再变化时没有明显代价，才值得进入真实多人 benchmark。
