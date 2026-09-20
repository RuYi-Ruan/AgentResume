# AgentResume

> 当前实验主线为 **M19**：把 M17 的严格配对协议放到 **9 组训练好的 Alice（3 seed × 3 地图）** 上复现，并补了 200 步合作得分续局。**9/9 组方向一致**——意图：旧印象 21.8% vs 新印象 90.0%；印象与真身"匹配 vs 不匹配" 84.4% vs 16.2%；得分：新−旧 +0.393 碗（p<1e-4，8/9 组正向）。完整结果见 [`M19_RESULTS.md`](docs/experiments/M19_RESULTS.md)。
>
> 历史脉络：M17 是最初的单组发现（[`final.md`](docs/experiments/final.md)）；M18 是早期确认性尝试，A 协议弃答失效、未复现（[`M18_FINAL_ANALYSIS.md`](docs/experiments/M18_FINAL_ANALYSIS.md)）；M15/M16 为更早探索。

AgentResume 是一个基于 Overcooked 的双智能体研究原型，用于研究“伙伴能力变化后，旧印象是否会导致意图误读”。

项目关注如下命题：Alice 通过经验积累提升能力后，其行为—意图映射会发生变化；如果 Bob 仍用过去的印象理解 Alice，就可能系统性误判她的意图；当 Bob 更新到与 Alice 一致的经验或能力模型后，判断应当恢复。

## 当前研究路线

当前主线使用同一个 `LLMPlayer` 类构造 Alice 和 Bob，并尽量只改变二者获得的上下文：

- `before`：Alice 没有经验库，Bob 不知道经验；
- `after_unaware`：Alice 注入经验库 E，Bob 仍按普通厨师理解 Alice；
- `after_aware`：Alice 注入 E，Bob 也持有 E，并模拟“带有这些经验的 Alice”后猜测意图。

LLM 只在决策点选择 `FETCH`、`PLACE`、`COOK_START`、`GET_DISH`、`PICKUP`、`DELIVER`、`PRE` 或 `HOLD` 等子目标。寻路、朝向和交互由确定性的电机层执行。智能体使用以自身为中心的 3×3 部分观测、锅状态记忆和延迟一 tick 的伙伴意图板。

经验库由无经验轨迹中的失败或低效片段挖掘得到。目前包含“空等时补料”和“ready 后及时取汤交付”等条目。

> 注意：仓库内 `data/v2_results_formal.json` 是决策事件计分修复前的探索性结果。旧实现会让 `guess` 和 `cause` 跨 tick 持续存在，因而可能重复计数；这些数值不能作为最终统计，需要使用修复后的代码重新运行。

## 历史实验

仓库保留了两代早期方案，便于复现研究演进，但不应与当前主线混为同一组结论：

1. 确定性状态机智能体：`CookAgent` 通过 `parallel_after_delay` 等显式参数表示串行/并行烹饪能力，行为稳定、便于统计，但能力提升由人工规则定义。
2. LLM + 状态机脚手架：LLM 在稀疏决策点选择子目标，`CookAgent` 同时提供角色、合法目标和默认策略，因此结果无法完全归因于 LLM。

当前的 `LLMPlayer` 去掉了状态机决策骨架，两名智能体使用同一实现，差异主要来自经验与印象上下文；容量限制、防堵和退让仍作为执行安全护栏存在。

## 目录结构

```text
src/ocres/
  grid.py          地图、功能位与环境构造
  recipes.py       锅状态、持有物与烹饪状态工具
  executor.py      寻路、朝向与原子交互执行层
  sensor.py        3×3 部分观测与锅状态记忆
  runner.py        逐 tick rollout 与日志
  llm.py           SiliconFlow/OpenAI 兼容接口
  llm_player.py    当前 LLM 智能体主线
  metrics.py       Alice 决策事件与 Bob 猜测的精确配对统计
  trainable.py     PyTorch 观测、数据集、动作掩码、策略与 checkpoint
  agents.py        历史确定性状态机智能体
  llm_agent.py     历史 LLM + 状态机脚手架

scripts/
  m7_collect_naive.py  采集无经验轨迹
  m7_mine_E.py         从轨迹挖掘经验库 E
  m9_formal.py         三条件正式实验
  m11_trainable_policy.py  PyTorch 宏观意图行为克隆/GPU 验证
  m12_closed_loop_bc.py    行为克隆策略与 L0 专家的同 seed 闭环对照
  m13_train_ppo.py         从 BC checkpoint 继续进行宏观动作 PPO 训练
  m14_checkpoint_impression.py  真实 pre/post checkpoint 的印象滞后反事实
  v2_debug.py          提示词与轨迹调试

tests/                 离线回归测试
data/                  轻量结果与经验库；原始 NPZ 轨迹不提交
```

## 环境安装

推荐 Windows 11、Python 3.11。当前开发环境使用独立 Conda 环境，避免与仓库旧 `.venv` 或 Conda base 混用：

```powershell
conda create -n agentresume python=3.11
conda activate agentresume
$env:PYTHONNOUSERSITE = "1"
python -m pip install -r requirements.txt
```

可训练策略阶段额外使用 PyTorch。当前机器安装的是 CUDA 11.8 版本：

```powershell
python -m pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

安装前用 `python -c "import sys; print(sys.executable)"` 确认真正的解释器位于 `C:\Users\86134\.conda\envs\agentresume`；提示符中的环境名称本身不足以证明解释器路径正确。

`overcooked-ai==1.1.0` 使用了 NumPy 2 已移除的接口，因此项目固定为 `numpy==1.26.4`，并显式安装其未声明的 SciPy 依赖。

## LLM 配置

默认接口为 SiliconFlow，默认模型为 `Qwen/Qwen3.5-9B`。可任选一种方式提供配置：

- 在仓库根目录创建不会被 Git 跟踪的 `apikey.txt`，第一行写 API key；
- 设置 `SILICONFLOW_API_KEY`、`QWEN_MODEL` 和 `QWEN_BASE_URL` 环境变量。

不要提交 API key。请求默认关闭模型思考模式，以避免结构化 JSON 输出被 reasoning token 挤占。

## 运行

PowerShell：

```powershell
$env:PYTHONPATH = "src"
python scripts\m0_env_check.py
python scripts\v2_debug.py
python scripts\m9_formal.py 1
```

完整数据流程：

```powershell
$env:PYTHONPATH = "src"
python scripts\m7_collect_naive.py
python scripts\m7_mine_E.py
python scripts\m9_formal.py 8
```

可训练策略与印象实验：

```powershell
$env:PYTHONNOUSERSITE = "1"
$env:MKL_THREADING_LAYER = "SEQUENTIAL"
$env:PYTHONPATH = "src"
python scripts\m11_trainable_policy.py --device cuda
python scripts\m12_closed_loop_bc.py --device cuda
python scripts\m13_train_ppo.py --device cuda
python scripts\m14_checkpoint_impression.py --device cuda
```

M13 使用 validation seeds 选择 checkpoint，并只在训练结束后报告独立 test seeds；M14 再使用另一组不重叠 seed 训练和测试 Bob 的意图模型。

调用远程模型可能耗时较长并产生 API 费用。建议先运行单局和离线测试：

```powershell
$env:PYTHONPATH = "src"
$env:MKL_THREADING_LAYER = "SEQUENTIAL"
python -m unittest discover -s tests -v
```

## 指标口径

主指标是 Bob 对 Alice 子目标的猜测准确率。修复后的事件协议为：

1. Alice 每次真正调用 LLM 决策时产生唯一 `decision_id`；
2. 该决定在下一 tick 发布到共享意图板；
3. Bob 针对该 ID 只产生一个 `guess_event`；
4. `metrics.score_intent_events` 按 ID 精确配对，不把跨 tick 保留的状态快照重复计数；
5. 另外统计 Alice 明确报告采用某条经验时的 `cause_accuracy` 子集。

吞吐量和交付分数属于辅助指标。经验注入可能改变决策，却未必稳定提高任务吞吐，因此不能仅凭分数认定能力提升。

## 下一步：可训练策略与“印象”表示

LLM 经验注入的优势是经验和印象可直接用自然语言表达，但策略变化不稳定。后续可引入可训练策略作为 Alice 的稳定能力载体，同时把“印象”建模为 Bob 对 Alice 隐变量的估计，而不是要求普通 MLP 直接理解文本：

- Alice 策略：`π_A(action | observation, capability)`，通过行为克隆、强化学习或课程学习得到不同能力 checkpoint；
- Bob 印象编码器：根据 Alice 的历史观测—动作序列输出 belief embedding 或能力层级分布；
- Bob 猜意头：`q(intent | current state, Alice action/history, belief)`；
- `stale` 条件冻结旧 belief，`updated` 条件用新轨迹更新 belief；
- 若必须输入自然语言经验，可先用冻结的文本编码器生成 embedding，再送入 MLP，而不是让 MLP处理 token。

这种设计把“真实能力”“Bob 的印象”和“意图预测”拆成可控变量，也能继续保留当前 LLM 方案作为语言型对照。

### RECOLLAB 对齐与首个 MLP 可行性实验

参考 Wallace 等人的 *ReCollab: Retrieval-Augmented LLMs for Cooperative Ad-hoc Teammate Modeling*，本项目进一步采用“伙伴类型 belief 与控制策略分离”的思路。RECOLLAB 本身并不让策略 MLP 读取自然语言印象：它先从 probe 轨迹预测离散伙伴类型，再路由到对应的预训练 best-response policy。

仓库已加入纯 NumPy 条件 MLP 原型 `scripts/m10_mlp_impression.py`。它在按 seed 留一的条件下，用“当前可观察局面 + L0/Lk impression one-hot”预测 Alice 宏观意图，并在同一 Lk 测试事件上反事实切换印象。当前历史数据上的均值为：

| 条件 | Lk 意图准确率 | reinterpretation 子集 |
|---|---:|---:|
| stale L0 impression | 0.821 | 0.000 |
| updated Lk impression | 1.000 | 1.000 |
| state-only，无 impression | 0.937 | 0.000 |

该结果只证明结构化 impression 能被 MLP 使用。其来源轨迹仍由历史状态机生成，而且不同 seed 高度同构，不能用来声称“训练产生了能力提升”。完整方案和有效性约束见 `docs/experiments/experiment_design_mlp.md`；下一阶段需要通过行为克隆初始化和 PPO/IPPO 训练得到真正的 pre/post Alice checkpoint。

## 文档

- [`docs/experiments/`](docs/experiments/README.md)：实验文档目录，包含最终总结、M14—M17 实验与早期探索记录；
- [`final.md`](docs/experiments/final.md)：当前实验配置、流程、结果和结论；
- [`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md)：项目交接与实现说明。

## License

本项目使用仓库根目录中的 `LICENSE`。
