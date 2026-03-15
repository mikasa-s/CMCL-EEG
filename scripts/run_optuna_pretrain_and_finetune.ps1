param(
    [string]$PretrainDatasets = "ds002336,ds002338,ds002739",
    [ValidateSet("ds002336", "ds002338", "ds002739")]
    [string]$TargetDataset = "ds002739",
    [string]$JointTrainConfig = "configs/train_joint_contrastive.yaml",
    [string]$FinetuneConfig = "configs/finetune_ds002739.yaml",
    [string]$OutputRoot = "outputs/optuna_run",
    [string]$CacheRoot = "cache",
    [double]$JointEegWindowSec = 8.0,
    [int]$PretrainEpochs = 0,
    [int]$FinetuneEpochs = 0,
    [int]$PretrainBatchSize = 0,
    [int]$FinetuneBatchSize = 0,
    [int]$BatchSize = 0,
    [int]$EvalBatchSize = 0,
    [int]$NumWorkers = -1,
    [switch]$SkipPretrain,
    [switch]$SkipFinetune,
    [switch]$TestOnly,
    [switch]$ForceCpu,
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))

$resolvedOutputRoot = if ([System.IO.Path]::IsPathRooted($OutputRoot)) {
    [System.IO.Path]::GetFullPath($OutputRoot)
}
else {
    [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputRoot))
}

$resolvedCacheRoot = if ([System.IO.Path]::IsPathRooted($CacheRoot)) {
    [System.IO.Path]::GetFullPath($CacheRoot)
}
else {
    [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $CacheRoot))
}

$jointCacheRoot = Join-Path $resolvedCacheRoot "joint_contrastive"
$ds002336CacheRoot = Join-Path $resolvedCacheRoot "ds002336"
$ds002338CacheRoot = Join-Path $resolvedCacheRoot "ds002338"
$ds002739CacheRoot = Join-Path $resolvedCacheRoot "ds002739"
$sharedOutputRoot = $resolvedOutputRoot

$cliParams = @(
    "-ExecutionPolicy", "Bypass",
    "-File", "scripts/run_pretrain_and_finetune.ps1",
    "-PretrainDatasets", $PretrainDatasets
)
$cliParams += @(
    "-TargetDataset", $TargetDataset,
    "-PythonExe", $PythonExe,
    "-JointTrainConfig", $JointTrainConfig,
    "-JointCacheRoot", $jointCacheRoot,
    "-Ds002336CacheRoot", $ds002336CacheRoot,
    "-Ds002338CacheRoot", $ds002338CacheRoot,
    "-Ds002739CacheRoot", $ds002739CacheRoot,
    "-JointOutputRoot", $sharedOutputRoot,
    "-Ds002336OutputRoot", $sharedOutputRoot,
    "-Ds002338OutputRoot", $sharedOutputRoot,
    "-Ds002739OutputRoot", $sharedOutputRoot,
    "-JointEegWindowSec", $JointEegWindowSec.ToString()
)

if ($TargetDataset -eq "ds002336") {
    $cliParams += @( "-Ds002336FinetuneConfig", $FinetuneConfig )
}
elseif ($TargetDataset -eq "ds002338") {
    $cliParams += @( "-Ds002338FinetuneConfig", $FinetuneConfig )
}
else {
    $cliParams += @( "-Ds002739FinetuneConfig", $FinetuneConfig )
}

if ($PretrainEpochs -gt 0) {
    $cliParams += @( "-PretrainEpochs", $PretrainEpochs.ToString() )
}
if ($FinetuneEpochs -gt 0) {
    $cliParams += @( "-FinetuneEpochs", $FinetuneEpochs.ToString() )
}
if ($BatchSize -gt 0) {
    $cliParams += @( "-BatchSize", $BatchSize.ToString() )
}
if ($PretrainBatchSize -gt 0) {
    $cliParams += @( "-PretrainBatchSize", $PretrainBatchSize.ToString() )
}
if ($FinetuneBatchSize -gt 0) {
    $cliParams += @( "-FinetuneBatchSize", $FinetuneBatchSize.ToString() )
}
if ($EvalBatchSize -gt 0) {
    $cliParams += @( "-EvalBatchSize", $EvalBatchSize.ToString() )
}
if ($NumWorkers -ge 0) {
    $cliParams += @( "-NumWorkers", $NumWorkers.ToString() )
}
if ($SkipPretrain) {
    $cliParams += "-SkipPretrain"
}
if ($SkipFinetune) {
    $cliParams += "-SkipFinetune"
}
if ($TestOnly) {
    $cliParams += "-TestOnly"
}
if ($ForceCpu) {
    $cliParams += "-ForceCpu"
}

& powershell @cliParams
exit $LASTEXITCODE