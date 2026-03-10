param(
    [string]$Config = "configs/train_ds002336.yaml"
)

$ErrorActionPreference = "Stop"

Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))

$python = "D:\anaconda3\envs\mamba\python.exe"

Write-Host "Running contrastive training for ds002336 binary block dataset..."
& $python run_train.py --config $Config