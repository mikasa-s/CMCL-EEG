param(
    [string]$Config = "configs/train_ds002739.yaml"
)

$ErrorActionPreference = "Stop"

Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))

$python = "D:\anaconda3\envs\mamba\python.exe"

Write-Host "Running contrastive training for ds002739 subject-packed dataset..."
& $python run_train.py --config $Config
