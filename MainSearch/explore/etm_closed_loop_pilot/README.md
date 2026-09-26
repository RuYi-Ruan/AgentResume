# ETM 闭环共同学习小实验

这个实验承接 `etm_gradual_learning_pilot`，检查动态 ETM 的正向信号在闭环共同学习中是否仍然存在。

与前一个实验的主要区别：

- A、B、C 三个固定身份智能体都持续学习；
- 每个智能体都维护对另外两人的队友模型；
- 三人的最终角色共同决定团队奖励；
- 团队奖励进入每个智能体的 Q 值更新，因此任何人的决策都会改变其他人的后续学习轨迹；
- 不切换 checkpoint，每轮只进行一次小幅在线更新。

每轮中，智能体先根据当前策略形成角色提议，并发出带噪声的行为线索；然后分别推断另外两人的提议，选择最终角色。三种角色全部被覆盖时协作成功。

比较 `no_model`、`frozen_etm`、`dynamic_etm` 和 `oracle` 四种条件。这里的 `oracle` 只表示准确知道另外两人的角色提议，并不是集中控制的团队收益上界；三人同时根据相同信息调整时仍可能产生冲突。因为不同条件会形成不同的后续训练数据，本实验只能使用相同种子和共同随机数进行配对，不能像前一个开环实验那样共享完全相同的伙伴轨迹。

运行：

```powershell
.\.venv\Scripts\python.exe MainSearch\explore\etm_closed_loop_pilot\run_experiment.py
```

测试：

```powershell
.\.venv\Scripts\python.exe -m unittest MainSearch.explore.etm_closed_loop_pilot.test_experiment
```

输出位于 `results/`。这是开发性玩具实验，不是正式 benchmark。
