# SMAX 官方 QMIX 短程可运行性测试

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-24
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1

## 目的

只检验 JaxMARL v0.2.0 官方 QMIX 在本机能否完成编译、8192 环境步训练、离线日志和模型保存，并测量耗时。**不以此判断胜率、ETM 或五套独立策略。**

## 安装与运行（PowerShell；每行单独执行）

```powershell
Set-Location -LiteralPath 'D:\omp\MainSearch\explore\benchmark_suitability_smax'
git clone --depth 1 --branch v0.2.0 https://github.com/FLAIROx/JaxMARL.git jaxmarl_ref
Set-Location -LiteralPath 'D:\omp\MainSearch\explore\benchmark_suitability_smax\jaxmarl_ref'
& 'D:\omp\MainSearch\explore\benchmark_suitability_smax\.venv\Scripts\python.exe' -m pip install -e '.[algs]' 'jax==0.4.36' 'jaxlib==0.4.36'
& 'D:\omp\MainSearch\explore\benchmark_suitability_smax\.venv\Scripts\python.exe' -m pip install 'flax==0.10.0' 'jax==0.4.36' 'jaxlib==0.4.36'
$env:WANDB_DIR = 'D:\omp\MainSearch\explore\benchmark_suitability_smax'
& 'D:\omp\MainSearch\explore\benchmark_suitability_smax\.venv\Scripts\python.exe' baselines/QLearning/qmix_rnn.py +alg=ql_rnn_smax alg.MAP_NAME=smacv2_5_units alg.TOTAL_TIMESTEPS=8192 alg.NUM_ENVS=2 alg.NUM_STEPS=32 alg.BUFFER_SIZE=512 alg.BUFFER_BATCH_SIZE=8 alg.HIDDEN_SIZE=64 alg.MIXER_EMBEDDING_DIM=16 alg.MIXER_HYPERNET_HIDDEN_DIM=32 alg.LEARNING_STARTS=512 alg.NUM_EPOCHS=1 alg.TEST_NUM_ENVS=16 alg.TEST_INTERVAL=0.2 WANDB_MODE=offline SAVE_PATH=D:/omp/MainSearch/explore/benchmark_suitability_smax/smoke_models
```

若仓库已存在，不要重复 `git clone`；先核查原目录来源和版本。不要把 8192 步的胜率当成学习成效。失败信息和完整输出均保留，不静默重试。

版本备注：JAX 0.4.36 搭配 Flax 0.10.4、0.8.5 的两次失败记录见 `RESULTS.md`；Flax 0.10.0 的第三次运行通过。

## 记录与判据

- 记录命令开始、结束时间与是否正常退出；查看离线 `wandb/` 日志是否包含 `env_step`、`test_returned_won_episode`，以及 `smoke_models/` 是否保存参数。
- 成功判据仅为脚本可运行、训练步数接近 8192、测试和保存完成。时间与内存开销决定是否继续做 10 万步试跑。
- 正式训练前还需解决五人独立策略、固定测试集和逐检查点日志；官方共享参数 QMIX 不满足这些条件。
