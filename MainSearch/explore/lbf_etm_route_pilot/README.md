# LBF动态ETM线路小实验

目标：快速验证“真实三智能体LBF + 渐进共同学习 + 动态ETM”这条线路能否跑通。

- 三名固定身份智能体持续在线更新，不切换checkpoint；
- 三个食物分别需要1、2、3名智能体合作，产生不同伙伴组合；
- 高层意图是目标食物，合作伙伴是选择同一目标的其他智能体；
- 伙伴level从决策输入中移除，三名智能体物理level固定为1；
- 比较无模型、冻结ETM和动态ETM；
- 低层移动使用固定规划器，高层选择和环境采集使用真实LBF动力学。

这是线路验证，不是正式方法实验。

运行：

```powershell
.\.venv\Scripts\python.exe MainSearch\explore\lbf_etm_route_pilot\run_experiment.py
```

测试：

```powershell
.\.venv\Scripts\python.exe -m unittest MainSearch.explore.lbf_etm_route_pilot.test_experiment
```
