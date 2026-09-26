param(
    [int]$Seed = 0,
    [switch]$CheckOnly,
    [switch]$Smoke
)

$ErrorActionPreference = 'Stop'
$python = 'C:\Users\86134\.conda\envs\agentresume\python.exe'
$reference = Join-Path $PSScriptRoot 'epymarl_ref'
$main = Join-Path $reference 'src\main.py'
$budget = 500000
$testInterval = 50000
$testEpisodes = 100
$output = Join-Path $PSScriptRoot "qmix_500k_seed$Seed"
if ($Smoke) {
    $budget = 150
    $testInterval = 100000
    $testEpisodes = 2
    $output = Join-Path $PSScriptRoot "qmix_500k_smoke_seed$Seed"
}

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Missing Python environment: $python"
}
if (-not (Test-Path -LiteralPath $main -PathType Leaf)) {
    throw "Missing EPyMARL entry point: $main"
}

& $python -c 'import torch, gymnasium, smaclite, sacred, rtree, yaml; print(torch.__version__, torch.cuda.is_available())'
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }

Write-Host "Reference: $reference"
Write-Host "Seed: $Seed; map: 2s3z; training budget: $budget environment steps"
Write-Host "Models: $output\models"
Write-Host "Metrics: $reference\results\sacred\qmix\2s3z"
if ($CheckOnly) { return }

$trainArgs = @(
    'src/main.py', '--config=qmix', '--env-config=smaclite', '-C', 'sys', 'with',
    "seed=$Seed", 'env_args.map_name=2s3z', "t_max=$budget",
    'epsilon_anneal_time=50000', 'use_rnn=False',
    "test_interval=$testInterval", "test_nepisode=$testEpisodes",
    'fixed_eval_starts=True', 'final_eval_and_save=True',
    'save_model=True', 'save_model_interval=50000',
    'use_cuda=False', 'name=qmix_500k',
    "local_results_path=$($output.Replace('\', '/'))"
)

Push-Location $reference
try {
    & $python @trainArgs
    if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE." }
} finally {
    Pop-Location
}
