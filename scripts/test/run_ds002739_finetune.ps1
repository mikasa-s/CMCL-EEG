param(
    [string]$Config = "configs/finetune_ds002739.yaml",
    [string]$ContrastiveCheckpoint = "",
    [string]$FinetuneCheckpoint = "",
    [switch]$TestOnly
)

$ErrorActionPreference = "Stop"

Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))

$python = "python"

Write-Host "Running finetuning for ds002739 subject-packed dataset..."
$commandLine = @("run_finetune.py", "--config", $Config)

if (-not [string]::IsNullOrWhiteSpace($ContrastiveCheckpoint)) {
    $commandLine += @("--contrastive-checkpoint", $ContrastiveCheckpoint)
}
if (-not [string]::IsNullOrWhiteSpace($FinetuneCheckpoint)) {
    $commandLine += @("--finetune-checkpoint", $FinetuneCheckpoint)
}
if ($TestOnly) {
    $commandLine += "--test-only"
}
if ($TestOnly) {
    Write-Host "Test-only mode enabled. The run will load an existing finetune checkpoint and only evaluate the test split."
}
elseif ([string]::IsNullOrWhiteSpace($ContrastiveCheckpoint)) {
    Write-Host "No contrastive checkpoint provided. Finetuning will use random initialization unless modality-specific checkpoints are configured."
}
& $python @commandLine