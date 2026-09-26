# RWARE 4 智能体学习门槛（探索）

目的：为本项目下一轮主实验候选环境 **Jumanji RobotWarehouse（`tiny-4ag`，4 个固定身份机器人）** 建立可复现的“每局配送数”学习曲线，并记录成本。这是环境与训练管线的门槛，不是 ETM 结果。

## 为什么是这里

见 `../benchmark_screening_2026_09/RESULTS.md`：本机实测原始环境 1.4–2.2 万环境步/秒，自建 IPPO 训练 5,418 环境步/秒（对比 OvercookedV2 CNN+GRU 管线 205 步/秒）。RWARE 具备 ETM 需要的结构：4 个固定身份、共享奖励、个体部分观测、配送数可分；且本项目此前用 EPyMARL + semitable RWARE 失败，正好用快得多的管线重检。

## 与参考实现的关系

- 场景与超参对齐 InstaDeep **Mava** 的 RWARE `tiny-4ag` 配置（rollout 128、4 PPO epoch、2 minibatch、LR 2.5e-4、clip 0.2、ent 0.01、gamma 0.99、gae 0.95、MLP [128,128]、agent-ID one-hot、time_limit 500）。
- Mava 本体**未能在 Windows 上安装**（`pytinyrenderer` 需要 MSVC）；因此训练器为自建单文件实现（JAX + Flax + optax，Jumanji 环境）。这一差异已记录，不冒充 Mava 复现。

## 文件与运行

- `train_rware_ippo.py`：分段 jitted 训练 + 固定局面冻结评测（greedy 与 sampled 两种动作模式）+ 检查点 + 逐段 JSON。
- `results/<run>/run.json`：配置、固定评测局面指纹、逐段指标与耗时；`checkpoint_*.pkl`：参数与优化器状态。

```powershell
& 'D:\omp\MainSearch\explore\benchmark_suitability_smax\.venv\Scripts\python.exe' train_rware_ippo.py `
  --num-envs 64 --rollout-length 128 --updates 1000 --segment-updates 50 --eval-episodes 32
```

## 记录口径与已知限制

- 指标：每局配送数（= 共享回报，1 次配送 +1）、每局长度（碰撞或非法动作即结束，上限 500 步）。
- 评测用固定 PRNG 键（32 局），同时报 greedy（argmax）与 sampled（采样）两种动作模式：欠训练策略的 argmax 可能退化为常量合法动作而“看起来不动”，采样模式更敏感（V2 校准的教训）。
- 当前版本四个智能体**共享参数**（标准 IPPO）。ETM 阶段需要四套独立参数策略，改造点集中在 actor/critic 调用处。
- 失败与负结果同样保留，不静默覆盖。
