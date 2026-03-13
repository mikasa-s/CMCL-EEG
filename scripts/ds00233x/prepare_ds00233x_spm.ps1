param(
    [ValidateSet("ds002336", "ds002338")]
    [string]$DatasetName = "ds002336",
    [string]$DsRoot = "../ds002336",
    [string]$OutputRoot = "cache/ds002336",
    [string[]]$Subjects = @(),
    [string[]]$Tasks = @(),
    [ValidateSet("none", "subject", "loso")]
    [string]$SplitMode = "loso",
    [int]$TrainSubjects = 14,
    [int]$ValSubjects = 2,
    [int]$TestSubjects = 1,
    [int]$EegSeqLen = 8,
    [int]$EegPatchLen = 200,
    [bool]$DropEcg = $true,
    [bool]$TrainingReady = $true,
    [bool]$EegOnly = $true,
    [ValidateRange(1, 64)]
    [int]$ParallelWorkers = 4,
    [string]$TargetChannelManifest = ""
)

$ErrorActionPreference = "Stop"

function Convert-ToMatlabCellArray {
    param([string[]]$Values)
    if (-not $Values -or $Values.Count -eq 0) {
        return "{}"
    }
    $escaped = foreach ($Value in $Values) {
        "'" + ($Value -replace "'", "''") + "'"
    }
    return "{" + ($escaped -join ",") + "}"
}

Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
$matlabScriptDir = Resolve-Path "preprocess"
$resolvedDsRoot = Resolve-Path $DsRoot

if (-not $Tasks -or $Tasks.Count -eq 0) {
    if ($DatasetName -eq "ds002336") {
        $Tasks = @("motorloc", "MIpre", "MIpost", "eegNF", "fmriNF", "eegfmriNF")
    }
    elseif ($DatasetName -eq "ds002338") {
        $Tasks = @("MIpre", "MIpost", "1dNF", "2dNF")
    }
}

$subjectsExpr = Convert-ToMatlabCellArray $Subjects
$tasksExpr = Convert-ToMatlabCellArray ($Tasks | ForEach-Object { "task-$_" })
$matlabCmd = "addpath('$matlabScriptDir'); run_spm_preproc_ds00233x($subjectsExpr,$tasksExpr,'$resolvedDsRoot',$ParallelWorkers);"

Write-Host ("Running MATLAB SPM preprocessing for " + $DatasetName + "...")
matlab -batch $matlabCmd
if ($LASTEXITCODE -ne 0) {
    throw ("MATLAB SPM preprocessing failed for " + $DatasetName)
}

Write-Host ("Running Python preprocessing for " + $DatasetName + "...")
$prepareArgs = @{
    DatasetName   = $DatasetName
    DsRoot        = $DsRoot
    OutputRoot    = $OutputRoot
    Tasks         = $Tasks
    SplitMode     = $SplitMode
    TrainSubjects = $TrainSubjects
    ValSubjects   = $ValSubjects
    TestSubjects  = $TestSubjects
    EegSeqLen     = $EegSeqLen
    EegPatchLen   = $EegPatchLen
    DropEcg       = $DropEcg
    TrainingReady = $TrainingReady
    EegOnly       = $EegOnly
    FmriSource    = "spm_smoothed"
}
if ($Subjects.Count -gt 0) {
    $prepareArgs["Subjects"] = $Subjects
}
if ($TargetChannelManifest.Trim()) {
    $prepareArgs["TargetChannelManifest"] = $TargetChannelManifest
}

& (Resolve-Path "scripts/ds00233x/prepare_ds00233x.ps1") @prepareArgs
exit $LASTEXITCODE
