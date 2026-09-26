param(
    [int]$Seed = 0,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$python = 'C:\Users\86134\.conda\envs\agentresume\python.exe'
$reference = Join-Path $PSScriptRoot 'epymarl_ref'
$main = Join-Path $reference 'src\main.py'
$output = Join-Path $PSScriptRoot "qmix_100k_seed$Seed"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Missing Python environment: $python"
}
if (-not (Test-Path -LiteralPath $main -PathType Leaf)) {
    throw "Missing EPyMARL entry point: $main"
}

& $python -c 'import torch, gymnasium, smaclite, sacred, rtree, yaml; print(torch.__version__, torch.cuda.is_available())'
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }

Write-Host "Reference: $reference"
Write-Host "Seed: $Seed; map: 2s3z; training budget: 100000 environment steps"
Write-Host "Models: $output\models"
Write-Host "Metrics: $reference\results\sacred\qmix\2s3z"
if ($CheckOnly) { return }

$trainArgs = @(
    'src/main.py', '--config=qmix', '--env-config=smaclite', '-C', 'sys', 'with',
    "seed=$Seed", 'env_args.map_name=2s3z', 't_max=100000',
    'epsilon_anneal_time=50000', 'use_rnn=False',
    'test_interval=20000', 'test_nepisode=100',
    'save_model=True', 'save_model_interval=20000',
    'use_cuda=False', 'name=qmix_100k',
    "local_results_path=$($output.Replace('\', '/'))"
)

Push-Location $reference
try {
    & $python @trainArgs
    if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE." }
} finally {
    Pop-Location
}
