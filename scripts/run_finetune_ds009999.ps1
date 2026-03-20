param(
    [string]$FinetuneConfig = "configs/finetune_ds009999.yaml",
    [string]$CacheRoot = "cache/ds009999",
    [string]$OutputRoot = "outputs/ds009999",
    [int]$Epochs = 0,
    [int]$BatchSize = 0,
    [int]$EvalBatchSize = 0,
    [int]$NumWorkers = -1,
    [string]$PythonExe = "",
    [switch]$ForceCpu
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))

function Read-JsonFile {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-Content -Path $Path -Raw | ConvertFrom-Json)
}

function Get-MetricValue {
    param([Parameter(Mandatory = $true)]$Metrics, [Parameter(Mandatory = $true)][string]$Name)
    if ($null -eq $Metrics.PSObject.Properties[$Name]) { return [double]::NaN }
    return [double]$Metrics.$Name
}

function Get-MeanValue {
    param([double[]]$Values)
    if ($null -eq $Values -or $Values.Count -eq 0) { return [double]::NaN }
    return ($Values | Measure-Object -Average).Average
}

function Get-StdValue {
    param([double[]]$Values)
    if ($null -eq $Values -or $Values.Count -le 1) { return 0.0 }
    $mean = Get-MeanValue -Values $Values
    $sum = 0.0
    foreach ($value in $Values) {
        $diff = $value - $mean
        $sum += $diff * $diff
    }
    return [math]::Sqrt($sum / ($Values.Count - 1))
}

function Write-FinetuneSummary {
    param([Parameter(Mandatory = $true)][string]$FinetuneRoot)

    $summaryRows = @()
    foreach ($foldDir in (Get-ChildItem -Path $FinetuneRoot -Directory | Where-Object { $_.Name -like "fold_*" } | Sort-Object Name)) {
        $metricsPath = Join-Path $foldDir.FullName "test_metrics.json"
        if (!(Test-Path $metricsPath)) {
            continue
        }
        $metrics = Read-JsonFile -Path $metricsPath
        $summaryRows += [pscustomobject]@{
            fold         = $foldDir.Name
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

    $summaryRows += [pscustomobject]@{
        fold         = "CROSS_FOLD_MEAN_STD"
        accuracy     = Get-MeanValue -Values @($summaryRows | ForEach-Object { [double]$_.accuracy })
        accuracy_std = Get-StdValue -Values @($summaryRows | ForEach-Object { [double]$_.accuracy })
        macro_f1     = Get-MeanValue -Values @($summaryRows | ForEach-Object { [double]$_.macro_f1 })
        macro_f1_std = Get-StdValue -Values @($summaryRows | ForEach-Object { [double]$_.macro_f1 })
        loss         = Get-MeanValue -Values @($summaryRows | ForEach-Object { [double]$_.loss })
    }

    $summaryPath = Join-Path $FinetuneRoot "loso_finetune_summary.csv"
    $summaryRows | Export-Csv -Path $summaryPath -NoTypeInformation -Encoding UTF8
}

$python = if ($PythonExe.Trim()) { $PythonExe } else { "python" }
$losoRoot = Join-Path $CacheRoot "loso_subjectwise"
if (!(Test-Path $losoRoot)) {
    throw "LOSO directory not found: $losoRoot"
}

$foldDirs = Get-ChildItem -Path $losoRoot -Directory | Where-Object { $_.Name -like "fold_*" } | Sort-Object Name
if ($foldDirs.Count -eq 0) {
    throw "No fold_* directories found under $losoRoot"
}

$finetuneRoot = Join-Path $OutputRoot "finetune"
New-Item -ItemType Directory -Force -Path $finetuneRoot | Out-Null

foreach ($foldDir in $foldDirs) {
    $argsList = @(
        "run_finetune.py",
        "--config", $FinetuneConfig,
        "--train-manifest", (Join-Path $foldDir.FullName "manifest_train.csv"),
        "--val-manifest", (Join-Path $foldDir.FullName "manifest_val.csv"),
        "--test-manifest", (Join-Path $foldDir.FullName "manifest_test.csv"),
        "--root-dir", $CacheRoot,
        "--output-dir", (Join-Path $finetuneRoot $foldDir.Name)
    )
    if ($Epochs -gt 0) {
        $argsList += @("--epochs", $Epochs.ToString())
    }
    if ($BatchSize -gt 0) {
        $argsList += @("--batch-size", $BatchSize.ToString())
    }
    if ($EvalBatchSize -gt 0) {
        $argsList += @("--eval-batch-size", $EvalBatchSize.ToString())
    }
    if ($NumWorkers -ge 0) {
        $argsList += @("--num-workers", $NumWorkers.ToString())
    }
    if ($ForceCpu) {
        $argsList += "--force-cpu"
    }
    Write-Host ("[" + $foldDir.Name + "] finetune")
    & $python @argsList
    if ($LASTEXITCODE -ne 0) {
        throw "Finetune failed on $($foldDir.Name) with exit code $LASTEXITCODE"
    }
}

Write-FinetuneSummary -FinetuneRoot $finetuneRoot
Write-Host ("Saved summary: " + (Join-Path $finetuneRoot "loso_finetune_summary.csv"))
