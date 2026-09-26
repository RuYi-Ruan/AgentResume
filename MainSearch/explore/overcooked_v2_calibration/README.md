# 双人 OvercookedV2 训练管线校准（探索）

目的：在**原生双人** OvercookedV2 地图上，用 JaxMARL v0.2.0 官方 CNN+GRU IPPO 实现，检查「带局内记忆、学习型原子动作」的训练管线能否在本机形成可复现的出汤学习曲线，并如实记录成本（墙钟时间、环境步、训练局数、检查点）。

这一步只校准管线，不是 ETM 结果，也不是对论文 3000 万步曲线的复现。双人只是最容易对齐参考实现的选择。背景见 `../search_line.md`（第 47–49 条）与 `../overcooked_v2_3p_gate/design_direction.md`。

## 参考代码与本地补丁

参考脚本：`../benchmark_suitability_smax/jaxmarl_ref/baselines/IPPO/ippo_rnn_overcooked_v2.py`（JaxMARL v0.2.0），带两处本地兼容补丁：

1. Flax 0.10 下把包装 CNN 的 `jax.vmap` 换成 `nn.vmap`（否则 `JaxTransformError`，训练 0 步）。
2. `train(rng, initial_runner_state=None)` 与可选 `RUN_UPDATES`，用于把训练切成可续跑的分段，便于在段间做检查点与冻结评测。

两处都是本地兼容改动，不当作官方原样复现。

## 文件

- `run_calibration.py`：分段训练 + 固定局面冻结评测 + 检查点 + 逐步指标 JSON。
- `smoke_calibration.py`：最短可用性校验（初始化、续跑、评测确定性、检查点往返、环境自动重置）。
- `measure_throughput.py`：几种候选配置的 CPU 吞吐（步/秒）与编译耗时。
- `results/`：原始输出，失败与负结果同样保留。

## 运行

```powershell
& 'D:\omp\MainSearch\explore\benchmark_suitability_smax\.venv\Scripts\python.exe' run_calibration.py --layout grounded_coord_simple --num-envs 32 --num-steps 64 --updates 64 --segment-updates 8
```

注意：必须使用上面的绝对路径解释器；相对路径 `.venv/Scripts/python.exe` 在本机 shell 里会解析到别的虚拟环境（无 JAX）。

## 记录口径

- 评测是**冻结固定局面**：固定 PRNG 键（初始菜谱与位置指纹写进 `results/<run>/run.json`），确定性 argmax 动作，400 步上限，报「每局正确出汤数」与形状化回报。
- 成本记实际环境步数（updates × num_envs × num_steps）、训练局数、墙钟时间与检查点文件。
- `--resume-from` 只恢复参数、优化器状态与更新计数；环境与 GRU 隐状态重新开始，这一偏差写在 `RESULTS.md`。
