# 第二轮 benchmark 检索（2026-09-24，用户要求：Hanabi 之外还有什么能用的）

## 为什么再检索

OvercookedV2 校准表明本机（仅 CPU）负担不起论文配方（单种子 41–74 小时），且可负担规模下没形成学习曲线。此前筛过的候选（SMAClite 2s3z、LBF、MPE2 Spread、VMAS、RWARE/EPyMARL、SMAX 8192 步）多因“训练管线太慢或胜率跳变”停在门槛前，Hanabi 用户已排除。因此本轮只问一个问题：**有没有本机能跑、有成熟公开学习基线、且真正需要伙伴认知的多人协作任务。**

网络限制记录：本机 DNS 屏蔽 `raw.githubusercontent.com`/`google.com`，搜索引擎（startpage/duckduckgo/ecosia/mojeek/google）全部不可用；可用的是 `github.com`（含 API 渲染）、`arxiv.org`、`pypi.org`、`openreview.net`。因此结论来自仓库/文档/PyPI 元数据直读，不是搜索引擎结果。

## 候选对照（含实测）

| 候选 | 智能体 | 本机可跑性 | 吞吐 | 公开学习基线 | ETM 契合 | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| OvercookedV2（官方 CNN+GRU IPPO） | 2（可扩 3） | 已装 | **205 环境步/秒**（实测，本次校准） | 论文 10 种子 × 3000 万步 | 高 | 本机不可负担，已判负 |
| **SMAX `3m`（JaxMARL IPPO+GRU）** | 3（另有 8m 等 14 图） | 已装，零改动 | **9,219 环境步/秒**（实测，含策略网络与 GRU 训练） | Mava 公开 11 个 SMAX 场景曲线（10 种子、TPE 调参）；SMAC 文献成熟 | 中（同质兵种、焦点火力） | **可立即跑 1000 万步级实验** |
| **Jumanji `RobotWarehouse-v0`（Mava RWARE）** | 4（Mava 场景 `tiny-4ag`） | 已装（纯 py 轮子） | **原始环境 14,181–21,757 环境步/秒**（两次实测：先 14,181 系本机并行其它任务时，后 21,757 为空闲时；均不含策略） | Mava 公开 15 个 RWARE 场景曲线（10 种子） | 中高（共享奖励、部分可观测、避碰+任务分配） | **首选候选** |
| JaxRobotarium（navigation / material_transport / arctic_transport / foraging / discovery / rware） | 可配 2–N | 未装（纯 JAX，需从 GitHub 装，并配 JaxMARL 集成） | 未测 | 论文（CoRL 2025）称分钟级训练，附 wall-time 图 | 中高（异构机器人：冰/水/无人机、不同速度与容量） | 备选，装完再测 |
| MPE2 / MPE（JaxMARL、Mava） | 3–6 | 已装 | 未测（小 obs，快） | Mava 公开 MPE 曲线 | 低（协调浅，Spread 已判负） | 不作为主候选 |
| MAgent2 | 数十 | Windows 有轮子 | 未测（非 JAX，C++） | 公开基线少 | 低 | 不优先 |
| PettingZoo multiwalker(3) / knights_archers_zombies(4) / waterworld(5) | 3–5 | Windows 可装（MuJoCo/pygame） | 未测（非 JAX，Python 步进慢） | SB3 基线 | 低-中（物理协调为主） | 不优先 |
| Melting Pot 2（协作烹饪 / clean_up / commons_harvest） | 3–16 | **不可用** | — | 论文成熟，ZSC 场景丰富 | 高 | `dmlab2d` 只发 linux/macos 轮子（PyPI 28 个文件，win=0），本机装不上；需 Linux/WSL/Docker |
| Neural MMO 3 / Gigastep | 大量 | 纯 py 可装 | 需 GPU 级规模 | 有 | 低（规模不匹配） | 不采用 |
| Hanabi（JaxMARL） | 2–5 | 已装 | 已试 | — | — | 用户已排除 |

## 关键发现

1. **吞吐差 45–70 倍**：本机 SMAX 训练 9,219 步/秒、Jumanji RWARE 原始环境 14,181 步/秒，而 V2 的 CNN+GRU 管线只有 205 步/秒。同样几小时预算，前者能跑 1,000 万–1 亿步，后者只能跑 100 万步。
2. **有现成的、经过调参且有公开曲线的训练系统**：InstaDeep Mava 提供 JAX 的 IPPO/MAPPO/QMIX/IQL/SAC/Sable 单文件实现，附带 `env=rware`（场景 `tiny-2ag, tiny-4ag, tiny-4ag-easy, small-4ag`）、`env=smax`、`env=lbf`、`env=mpe`、`env=mabrax`、`env=matrax` 的 Hydra 配置，并公开 15 个 RWARE / 11 个 SMAX / 7 个 LBF 场景的基准曲线（Sable 论文，10 种子）。这正是 V2 缺的东西：可靠、可复现的学习门槛。
3. **RWARE 与项目已有工作直接接续**：本项目此前用 EPyMARL + semitable RWARE（PyTorch，10 万步 ~40 秒……即极慢）失败；Jumanji 版是同一任务的 JAX 向量化实现（同一篇论文 `semitable/robotic-warehouse` 为原始来源），因此“任务无学习信号”的怀疑可以用快 100 倍的管线重新检验。
4. **ETM 契合度**：RWARE 4 个固定身份机器人、共享奖励、个体部分观测、配送数（shelves delivered）是清晰可分主指标；“伙伴能力随训练变化”天然成立（伙伴策略在线更新）。SMAX 3 个身份、胜率指标，适合做焦点火力式的伙伴意图预测。JaxRobotarium 的异构机器人（冰/水/无人机、不同速度与容量）最贴近“伙伴能力差异”的原始动机。

## 本机改动记录（不静默）

- 在共享环境 `benchmark_suitability_smax/.venv` 中新增安装：`jumanji 1.1.2`、`dm-env 1.6`、`esquilax 2.1.0`、`jax-tqdm 0.4.0`、`tqdm`（均为 `--no-deps`，JAX 仍为 0.4.36）。依赖链是逐个补齐的：先缺 `dm_env`，再缺 `esquilax`（`search_and_rescue` 导入），再缺 `jax_tqdm`。
- **Mava 全量安装在 Windows 上失败**（独立环境 `.venv_mava`，2026-09-24）：`pip install -e mava_ref` 在构建 `pytinyrenderer`（`rware`/`lbforaging` 一系渲染依赖）时需要 MSVC C++ 生成工具，本机没有 → `error: Microsoft Visual C++ 14.0 or greater is required`。其余重依赖（jax 0.5.3、gigastep、smaclite、id-marl-eval、jaxmarl fork、gym、rliable、type_enforced）都能构建成功，失败点只在 `pytinyrenderer`。
- 因此改为**自建单文件 IPPO 训练器**（`../rware_4p_gate/train_rware_ippo.py`）：Jumanji RWARE + 自有 JAX/Flax/optax 实现，超参与场景对齐 Mava 的 `tiny-4ag` 配置（rollout 128、4 epoch、2 minibatch、LR 2.5e-4、clip 0.2、ent 0.01、MLP[128,128]、agent-ID one-hot、time_limit 500）。实测 **5,418 环境步/秒**（64 环境×128 步，含策略与 PPO 更新），8.2M 步约 25–30 分钟。
- 若之后确实需要 Mava 原版代码：可尝试 `--no-deps` 安装 Mava 并手工补齐依赖（跳过需要编译的渲染包），本机未尝试；也可在 WSL/Linux 下安装。

## Mava 公开参考曲线（用于对照，不当作本机结果）

从 `docs/images/benchmark_results/rware.png`（仓库内图片，经读图工具解析）读取：横轴 timesteps 0–20M，纵轴 “Mean episode return”；四条曲线（无图例）终点约 **0.82 / 0.70 / 0.51 / 0.15**，均在 0 起步。该图是 **15 个 RWARE 场景的聚合**，因此绝对数值低，不能直接与本项目单场景（tiny-4ag）的“每局配送数”比较；它的用处是给出量级预期：**即便是调过参的基线，RWARE 也需要数百万到 2000 万步才能形成可见曲线**。本机 5,418 步/秒 → 2000 万步约 60 分钟，仍属可负担。

## 门槛进展

- **用户选择先跑 RWARE tiny-4ag**（2026-09-24）。已自建单文件 IPPO 训练器 `../rware_4p_gate/train_rware_ippo.py`，含固定局面冻结评测（greedy + sampled）与检查点；首跑 1000 次更新 = 8.19M 环境步、20 段、实测 5,418 环境步/秒，结果与判定写入 `../rware_4p_gate/RESULTS.md`。
- 仍待做（若 RWARE 门槛通过）：把共享参数策略改成 4 套固定身份独立策略，并验证伙伴行为随训练变化。
- 备选顺序不变：SMAX `3m`（已装、9,219 步/秒）→ JaxRobotarium（需安装并实测）。

原始数据：`results/smax_throughput.json`（本轮实测）、`results/rware_env_probe.txt`（环境接口与吞吐）、`results/mava_rware_benchmark.png`（Mava 公开参考曲线）、`../overcooked_v2_calibration/results/throughput.json`（V2 对照）。
