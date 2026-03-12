param(
    [ValidateSet("ds002336", "ds002739")]
    [string[]]$PretrainDatasets = @("ds002336", "ds002739"),
    [ValidateSet("ds002336", "ds002739")]
    [string]$TargetDataset = "ds002739",
    [string]$JointTrainConfig = "configs/train_joint_contrastive.yaml",
    [string]$FinetuneConfig = "configs/finetune_ds002739.yaml",
    [string]$OutputRoot = "outputs/optuna_run",
    [double]$JointEegWindowSec = 8.0,
    [int]$PretrainEpochs = 0,
    [int]$FinetuneEpochs = 0,
    [int]$BatchSize = 0,
    [int]$EvalBatchSize = 0,
    [int]$NumWorkers = -1,
    [switch]$SkipPretrain,
    [switch]$SkipFinetune,
    [switch]$TestOnly,
    [switch]$ForceCpu
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))

$resolvedOutputRoot = if ([System.IO.Path]::IsPathRooted($OutputRoot)) {
    [System.IO.Path]::GetFullPath($OutputRoot)
}
else {
    [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputRoot))
}

$jointCacheRoot = Join-Path $resolvedOutputRoot "cache\joint_contrastive"
$ds002336CacheRoot = Join-Path $resolvedOutputRoot "cache\ds002336"
$ds002739CacheRoot = Join-Path $resolvedOutputRoot "cache\ds002739"
$sharedOutputRoot = $resolvedOutputRoot

$args = @(
    "-ExecutionPolicy", "Bypass",
    "-File", "scripts/run_pretrain_and_finetune.ps1",
    "-PretrainDatasets"
)
$args += $PretrainDatasets
$args += @(
    "-TargetDataset", $TargetDataset,
    "-JointTrainConfig", $JointTrainConfig,
    "-JointCacheRoot", $jointCacheRoot,
    "-Ds002336CacheRoot", $ds002336CacheRoot,
    "-Ds002739CacheRoot", $ds002739CacheRoot,
    "-JointOutputRoot", $sharedOutputRoot,
    "-Ds002336OutputRoot", $sharedOutputRoot,
    "-Ds002739OutputRoot", $sharedOutputRoot,
    "-JointEegWindowSec", $JointEegWindowSec.ToString()
)

if ($TargetDataset -eq "ds002336") {
    $args += @( "-Ds002336FinetuneConfig", $FinetuneConfig )
}
else {
    $args += @( "-Ds002739FinetuneConfig", $FinetuneConfig )
}

if ($PretrainEpochs -gt 0) {
    $args += @( "-PretrainEpochs", $PretrainEpochs.ToString() )
}
if ($FinetuneEpochs -gt 0) {
    $args += @( "-FinetuneEpochs", $FinetuneEpochs.ToString() )
}
if ($BatchSize -gt 0) {
    $args += @( "-BatchSize", $BatchSize.ToString() )
}
if ($EvalBatchSize -gt 0) {
    $args += @( "-EvalBatchSize", $EvalBatchSize.ToString() )
}
if ($NumWorkers -ge 0) {
    $args += @( "-NumWorkers", $NumWorkers.ToString() )
}
if ($SkipPretrain) {
    $args += "-SkipPretrain"
}
if ($SkipFinetune) {
    $args += "-SkipFinetune"
}
if ($TestOnly) {
    $args += "-TestOnly"
}
if ($ForceCpu) {
    $args += "-ForceCpu"
}

& powershell @args
exit $LASTEXITCODE