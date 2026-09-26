# LBF 学习型低层动作门槛：开发实验

环境为官方 LBF `Foraging-5x5-3p-2f-coop-v3`。三名固定身份智能体各有独立 MLP，仅输入各自局部观测并直接输出 6 种原子动作；集中价值网络只在训练时使用。没有规则执行器、没有切换 checkpoint、没有 ETM。训练中使用基于三人到共同食物距离的势函数引导，评测只计环境原始奖励。

- 1 个种子，60 次更新 × 每次 24 局 = 1,440 局、72,000 环境步。
- 训练的 1,440 局中，仅 1 局获得正的原始采集奖励。
- 初始及每 10 次更新均在同一批 100 个测试种子评测；7 个检查点全部为 0/100 局获得正奖励。

结论：当前从零开始的集中训练/分散执行 PPO 没有通过能力渐进提升门槛。主要瓶颈是严格合作模式的成功轨迹太稀少，而非已证实 LBF 场景不适合 ETM。保留此负结果；下一步不应单纯延长同一配置训练，而应使用有参考实现的成熟多智能体训练基线，或明确记录一种训练期示范/课程学习初始化，并验证其后仍有真实在线能力增长。

运行命令：

```powershell
C:\Users\86134\.conda\envs\agentresume\python.exe MainSearch/explore/lbf_learned_actions_gate/train.py --updates 60 --episodes-per-update 24 --eval-episodes 100 --eval-interval 10 --output-dir MainSearch/explore/lbf_learned_actions_gate/development_seed0
```

原始数据在 `development_seed0/training.csv`、`evaluations.csv`、`summary.json`。
