param(
    [ValidateSet("ds002336", "ds002739")]
    [string[]]$PretrainDatasets = @("ds002336", "ds002739"),
    [ValidateSet("ds002336", "ds002739")]
    [string]$TargetDataset = "ds002739",
    [string]$JointTrainConfig = "configs/train_joint_contrastive.yaml",
    [string]$Ds002336FinetuneConfig = "configs/finetune_ds002336.yaml",
    [string]$Ds002739FinetuneConfig = "configs/finetune_ds002739.yaml",
    [string]$Ds002336Root = "../ds002336",
    [string]$Ds002739Root = "../ds002739",
    [string]$JointCacheRoot = "cache/joint_contrastive",
    [string]$Ds002336CacheRoot = "cache/ds002336",
    [string]$Ds002739CacheRoot = "cache/ds002739",
    [string]$JointOutputRoot = "outputs/joint_contrastive",
    [string]$Ds002336OutputRoot = "outputs/ds002336",
    [string]$Ds002739OutputRoot = "outputs/ds002739",
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

function Invoke-CommandOrThrow {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Args,
        [Parameter(Mandatory = $true)][string]$StepName
    )

    & $Executable @Args
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE"
    }
}

function Get-MetricValue {
    param(
        [Parameter(Mandatory = $true)]$Metrics,
        [Parameter(Mandatory = $true)][string]$Name
    )

    $property = $Metrics.PSObject.Properties[$Name]
    if ($null -eq $property -or $null -eq $property.Value) {
        return 0.0
    }
    return [double]$property.Value
}

function Get-MeanValue {
    param([double[]]$Values)
    if ($null -eq $Values -or $Values.Count -eq 0) {
        return 0.0
    }
    return [double](($Values | Measure-Object -Average).Average)
}

function Get-StdValue {
    param([double[]]$Values)
    if ($null -eq $Values -or $Values.Count -le 1) {
        return 0.0
    }
    $mean = Get-MeanValue -Values $Values
    $sum = 0.0
    foreach ($value in $Values) {
        $sum += [math]::Pow(([double]$value - $mean), 2)
    }
    return [math]::Sqrt($sum / $Values.Count)
}

function Read-JsonFile {
    param([Parameter(Mandatory = $true)][string]$Path)
    return Get-Content -Path $Path -Raw | ConvertFrom-Json
}

function Write-FinetuneSummary {
    param(
        [Parameter(Mandatory = $true)][string]$FinetuneRoot,
        [Parameter(Mandatory = $true)]$FoldNameMap
    )

    $summaryRows = @()
    foreach ($foldDir in (Get-ChildItem -Path $FinetuneRoot -Directory | Where-Object { $_.Name -like "fold_*" } | Sort-Object Name)) {
        $metricsPath = Join-Path $foldDir.FullName "test_metrics.json"
        if (!(Test-Path $metricsPath)) {
            continue
        }
        $metrics = Read-JsonFile -Path $metricsPath
        $summaryRows += [pscustomobject]@{
            fold         = $FoldNameMap[$foldDir.Name]
            fold_dir     = $foldDir.Name
            accuracy     = Get-MetricValue -Metrics $metrics -Name "accuracy"
            accuracy_std = Get-MetricValue -Metrics $metrics -Name "accuracy_std"
            macro_f1     = Get-MetricValue -Metrics $metrics -Name "macro_f1"
            macro_f1_std = Get-MetricValue -Metrics $metrics -Name "macro_f1_std"
            loss         = Get-MetricValue -Metrics $metrics -Name "loss"
        }
    }

    if ($summaryRows.Count -eq 0) {
        throw "No fold test_metrics.json files found under $FinetuneRoot"
    }

    $accuracyValues = @($summaryRows | ForEach-Object { [double]$_.accuracy })
    $macroF1Values = @($summaryRows | ForEach-Object { [double]$_.macro_f1 })
    $lossValues = @($summaryRows | ForEach-Object { [double]$_.loss })

    $summaryRows += [pscustomobject]@{
        fold         = "CROSS_FOLD_MEAN_STD"
        fold_dir     = ""
        accuracy     = Get-MeanValue -Values $accuracyValues
        accuracy_std = Get-StdValue -Values $accuracyValues
        macro_f1     = Get-MeanValue -Values $macroF1Values
        macro_f1_std = Get-StdValue -Values $macroF1Values
        loss         = Get-MeanValue -Values $lossValues
    }

    $summaryPath = Join-Path $FinetuneRoot "loso_finetune_summary.csv"
    $summaryRows | Export-Csv -Path $summaryPath -NoTypeInformation -Encoding UTF8
    $summaryRows | Format-Table -AutoSize | Out-String | Write-Host
}

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))

if ($SkipPretrain -and $SkipFinetune) {
    throw "SkipPretrain and SkipFinetune cannot both be set"
}

if ($TestOnly -and $SkipFinetune) {
    throw "TestOnly requires the finetune stage and cannot be combined with SkipFinetune"
}

$python = if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "python" }
$jointManifestPath = Join-Path $JointCacheRoot "manifest_all.csv"
$jointChannelManifest = Join-Path $JointCacheRoot "eeg_channels_target.csv"
$jointCheckpointPath = Join-Path $JointOutputRoot "contrastive\checkpoints\best.pth"

if (!$SkipPretrain) {
    $jointPrepareArgs = @(
        "-ExecutionPolicy", "Bypass",
        "-File", "scripts/prepare_joint_contrastive.ps1",
        "-OutputRoot", $JointCacheRoot,
        "-EegWindowSec", $JointEegWindowSec.ToString()
    )
    if ($PretrainDatasets.Count -gt 0) {
        $jointPrepareArgs += "-Datasets"
        $jointPrepareArgs += $PretrainDatasets
    }
    $jointPrepareArgs += @("-Ds002336Root", $Ds002336Root, "-Ds002739Root", $Ds002739Root)

    Write-Host "Preparing joint contrastive cache..."
    Invoke-CommandOrThrow -Executable "powershell" -Args $jointPrepareArgs -StepName "joint preprocessing"

    $trainArgs = @(
        "run_train.py",
        "--config", $JointTrainConfig,
        "--manifest", $jointManifestPath,
        "--root-dir", $JointCacheRoot,
        "--output-dir", (Join-Path $JointOutputRoot "contrastive")
    )
    if ($PretrainEpochs -gt 0) {
        $trainArgs += @("--epochs", $PretrainEpochs.ToString())
    }
    if ($BatchSize -gt 0) {
        $trainArgs += @("--batch-size", $BatchSize.ToString())
    }
    if ($NumWorkers -ge 0) {
        $trainArgs += @("--num-workers", $NumWorkers.ToString())
    }
    if ($ForceCpu) {
        $trainArgs += "--force-cpu"
    }

    Write-Host "Running joint contrastive pretraining..."
    Invoke-CommandOrThrow -Executable $python -Args $trainArgs -StepName "joint pretraining"
}

if ($SkipFinetune) {
    return
}

$targetPrepareScript = if ($TargetDataset -eq "ds002336") { "scripts/ds002336/prepare_ds002336.ps1" } else { "scripts/ds002739/prepare_ds002739.ps1" }
$targetFinetuneConfig = if ($TargetDataset -eq "ds002336") { $Ds002336FinetuneConfig } else { $Ds002739FinetuneConfig }
$targetCacheRoot = if ($TargetDataset -eq "ds002336") { $Ds002336CacheRoot } else { $Ds002739CacheRoot }
$targetOutputRoot = if ($TargetDataset -eq "ds002336") { $Ds002336OutputRoot } else { $Ds002739OutputRoot }

$targetPrepareArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $targetPrepareScript,
    "-OutputRoot", $targetCacheRoot,
    "-SplitMode", "loso",
    "-TrainingReady:$true"
)
if (Test-Path $jointChannelManifest) {
    $targetPrepareArgs += @("-TargetChannelManifest", $jointChannelManifest)
}
if ($TargetDataset -eq "ds002336") {
    $targetPrepareArgs += @("-DsRoot", $Ds002336Root)
}
else {
    $targetPrepareArgs += @("-DsRoot", $Ds002739Root)
}

Write-Host ("Preparing target finetune cache for " + $TargetDataset + "...")
Invoke-CommandOrThrow -Executable "powershell" -Args $targetPrepareArgs -StepName "target preprocessing"

$losoDir = Join-Path $targetCacheRoot "loso_subjectwise"
$foldDirs = Get-ChildItem -Path $losoDir -Directory | Where-Object { $_.Name -like "fold_*" } | Sort-Object Name
if ($foldDirs.Count -eq 0) {
    throw "No LOSO fold directories found under $losoDir"
}

$finetuneRoot = Join-Path $targetOutputRoot "finetune"
New-Item -ItemType Directory -Force -Path $finetuneRoot | Out-Null

$foldNameMap = @{}
for ($foldIndex = 0; $foldIndex -lt $foldDirs.Count; $foldIndex++) {
    $foldNameMap[$foldDirs[$foldIndex].Name] = "fold$($foldIndex + 1)"
}

foreach ($foldDir in $foldDirs) {
    $foldName = $foldDir.Name
    $trainManifest = Join-Path $foldDir.FullName "manifest_train.csv"
    $valManifest = Join-Path $foldDir.FullName "manifest_val.csv"
    $testManifest = Join-Path $foldDir.FullName "manifest_test.csv"
    $foldOutputDir = Join-Path $finetuneRoot $foldName
    $finetuneCheckpoint = Join-Path $foldOutputDir "checkpoints\best.pth"
    if (!(Test-Path $trainManifest) -or !(Test-Path $valManifest) -or !(Test-Path $testManifest)) {
        throw "Missing LOSO manifest(s) under $($foldDir.FullName)"
    }

    $finetuneArgs = @(
        "run_finetune.py",
        "--config", $targetFinetuneConfig,
        "--train-manifest", $trainManifest,
        "--val-manifest", $valManifest,
        "--test-manifest", $testManifest,
        "--root-dir", $targetCacheRoot,
        "--output-dir", $foldOutputDir
    )
    if (Test-Path $jointCheckpointPath) {
        $finetuneArgs += @("--contrastive-checkpoint", $jointCheckpointPath)
    }
    if ($TestOnly) {
        if (!(Test-Path $finetuneCheckpoint)) {
            throw "Expected finetune checkpoint not found for test-only: $finetuneCheckpoint"
        }
        $finetuneArgs += @("--finetune-checkpoint", $finetuneCheckpoint, "--test-only")
    }
    if ($FinetuneEpochs -gt 0) {
        $finetuneArgs += @("--epochs", $FinetuneEpochs.ToString())
    }
    if ($BatchSize -gt 0) {
        $finetuneArgs += @("--batch-size", $BatchSize.ToString())
    }
    if ($EvalBatchSize -gt 0) {
        $finetuneArgs += @("--eval-batch-size", $EvalBatchSize.ToString())
    }
    if ($NumWorkers -ge 0) {
        $finetuneArgs += @("--num-workers", $NumWorkers.ToString())
    }
    if ($ForceCpu) {
        $finetuneArgs += "--force-cpu"
    }

    Write-Host ("[" + $foldNameMap[$foldName] + "] finetune")
    Invoke-CommandOrThrow -Executable $python -Args $finetuneArgs -StepName ("finetune " + $foldName)
}

Write-FinetuneSummary -FinetuneRoot $finetuneRoot -FoldNameMap $foldNameMap