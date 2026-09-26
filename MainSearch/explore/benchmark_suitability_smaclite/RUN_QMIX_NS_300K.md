# `2s3z` 非共享 QMIX 门槛实验（已完成）

目的：检验五名固定身份智能体在同一批对局中、各用一套网络从零连续学习，能否形成可用协作；这不是 ETM 效果实验。

- 一次训练，seed 0，最多 300,000 环境步；不中途切换伙伴模型或重启训练。
- 每约 20,000 步用同一批 50 个评测种子测胜率和平均回报，并保存模型；终点强制评测、保存。
- 重点看 100,000、200,000、300,000 步的曲线。保存的检查点仅用于事后分析五人的行为变化，不在训练对局中切换。
- 这是单种子开发门槛，不作正式统计结论。无论成功、失败或中断，保留原始日志。

在 `D:\omp` 的 PowerShell 中运行：

```powershell
Set-Location -LiteralPath 'D:\omp\MainSearch\explore\benchmark_suitability_smaclite\epymarl_ref'
& 'C:\Users\86134\.conda\envs\agentresume\python.exe' src/main.py --config=qmix_ns --env-config=smaclite -C sys with seed=0 env_args.map_name=2s3z t_max=300000 epsilon_anneal_time=50000 test_interval=20000 test_nepisode=50 fixed_eval_starts=True final_eval_and_save=True save_model=True save_model_interval=20000 use_cuda=False name=qmix_ns_300k local_results_path=D:/omp/MainSearch/explore/benchmark_suitability_smaclite/qmix_ns_300k_seed0
```

当前 `agentresume` 环境为 CPU 版 PyTorch；不要把 `use_cuda` 改成 `True`。模型输出到 `qmix_ns_300k_seed0/models/`，Sacred 指标输出到 `epymarl_ref/results/sacred/qmix_ns/2s3z/` 的新运行编号。预计耗时不能从共享网络训练线性推算。

执行结果：Sacred run 1 已完成，共 300,017 环境步；终点评测 50/50 胜，但训练中出现剧烈波动。后续离线诊断见 `qmix_ns_checkpoint_diagnostic/RESULTS.md`。
