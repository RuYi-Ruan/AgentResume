# AgentResume

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
  agents.py        历史确定性状态机智能体
  llm_agent.py     历史 LLM + 状态机脚手架

scripts/
  m7_collect_naive.py  采集无经验轨迹
  m7_mine_E.py         从轨迹挖掘经验库 E
  m9_formal.py         三条件正式实验
  v2_debug.py          提示词与轨迹调试

tests/                 离线回归测试
data/                  轻量结果与经验库；原始 NPZ 轨迹不提交
```

## 环境安装

推荐 Windows 11、Python 3.11。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

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
.\.venv\Scripts\python.exe scripts\m0_env_check.py
.\.venv\Scripts\python.exe scripts\v2_debug.py
.\.venv\Scripts\python.exe scripts\m9_formal.py 1
```

完整数据流程：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\m7_collect_naive.py
.\.venv\Scripts\python.exe scripts\m7_mine_E.py
.\.venv\Scripts\python.exe scripts\m9_formal.py 8
```

调用远程模型可能耗时较长并产生 API 费用。建议先运行单局和离线测试：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
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

## 文档

- `PROJECT_HANDOFF.md`：当前实现与已知问题；
- `experiment_design_v2.md`：LLM 原生实验设计；
- `plan_motivation_validation.md`：整体实验规划与历史记录；
- `motivation_evidence.md`：已有探索性实验记录。

## License

本项目使用仓库根目录中的 `LICENSE`。
