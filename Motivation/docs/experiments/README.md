# 实验文档目录

本目录集中保存 AgentResume 的实验设计、阶段记录与结果解释。数据仍在仓库根目录的 `data/`，代码仍在 `scripts/` 与 `src/`。

阅读建议：

**M18 当前执行入口：[`M18_EXECUTION_PLAN_REVIEW.md`](M18_EXECUTION_PLAN_REVIEW.md)**。包含三张地图、训练与 seed 配置、全意图 Bob、采样/评分/预算和已获批准的五项选择；实际运行情况见 [`M18_PROGRESS.md`](M18_PROGRESS.md)。正式 Qwen 结果尚未产生。

严格配对自然来源不足后的 A 受控来源修订，见 [`M18_PAIR_SAMPLING_REVISION_REVIEW.md`](M18_PAIR_SAMPLING_REVISION_REVIEW.md)；D 自然事件容量不足后的采样修订，见 [`M18_D_SAMPLING_REVISION_REVIEW.md`](M18_D_SAMPLING_REVISION_REVIEW.md)。两项均已获用户批准，正式评估仍以各组能力与数据门槛为前提。

D-v2 的固定100局已经在九组采完，其中 B 地图三组均不足50；用户已批准 [`M18_D_CAPACITY_FOLLOWUP_REVIEW.md`](M18_D_CAPACITY_FOLLOWUP_REVIEW.md) 的“每组最多50、容量不足保留实际数”，D-v3 只新建接受凭据，不覆盖已失败的 D-v2 记录。Bob 的避让型控制器开发失败记录全部保留；最终冻结的 v5 改为流水线分工，在九组144个开发起点上 Oracle 的前300步平均+0.1667份汤并通过公开触发门槛，正式团队得分使用 v5。

1. [`M18_CONFIRMATORY_MOTIVATION_DESIGN_REVIEW.md`](M18_CONFIRMATORY_MOTIVATION_DESIGN_REVIEW.md)：M18 设计演进的早期版本，不以它代替最新执行方案；
2. [`final.md`](final.md)：当前 M17 实验的完整配置、流程、结果和结论边界；
3. [`M17_PROGRESS.md`](M17_PROGRESS.md)：M17 逐阶段进度与原始结果索引；
4. [`M17_EXPERIMENT_DESIGN_REVIEW.md`](M17_EXPERIMENT_DESIGN_REVIEW.md)：已审核的实验方案及执行时修订；
5. [`CODE_AND_EXPERIMENT_REVIEW_2026-09-11.md`](CODE_AND_EXPERIMENT_REVIEW_2026-09-11.md)：早期代码与实验复核；
6. [`RESEARCH_STATUS_AND_ROADMAP.md`](RESEARCH_STATUS_AND_ROADMAP.md)：研究路线的演进与证据边界。

按阶段查找：

- M16 与 M16-V2：`M16_*`、`M16_V2_*`，记录地图、Alice 训练和 Bob 观察方案的迭代；
- M15：`M15_EXPERIMENT_DESIGN.md`、`M15_RESULTS_EXPLANATION.md`，记录动作输入过于充分及合作得分未改善的问题；
- M14：`M14_DATA_EXPLANATION.md`，解释早期结构化 impression 的统计结果；
- 更早的探索：`experiment_design_*.md`、`plan_motivation_validation.md`、`motivation_evidence.md`。

文中以反引号写出的 `data/...`、`scripts/...` 等路径均相对于仓库根目录。早期结果不要与 M17 的正式统计混用。
