# LBF渐进共同学习门槛实验

目的：在加入ETM之前，验证真实三智能体LBF环境能否产生固定身份智能体通过持续经验逐步改善协作的轨迹。

## 设置

- 环境：`Foraging-8x8-3p-3f-coop-v3`；
- 三名智能体身份固定，不切换checkpoint；
- 每局每名智能体独立选择一个食物目标，三人选择相同目标时才执行合作采集；
- 每名智能体使用独立的在线Q值，通过共享环境奖励逐局更新；
- 低层移动使用固定联合路径规划器，避免把路径规划失败混入高层共同学习；
- 高层意图标签为目标食物，合作伙伴标签为另外两名智能体。

这是benchmark接入门槛，不是ETM效果实验。当前原始LBF观测包含level信息；正式ETM实验必须使用去level观测，避免直接读取能力。

## 运行

```powershell
.\.venv\Scripts\python.exe MainSearch\explore\lbf_gradual_learning_gate\run_experiment.py
```

```powershell
.\.venv\Scripts\python.exe -m unittest MainSearch.explore.lbf_gradual_learning_gate.test_experiment
```

结果写入 `results/`。
