param(
    [string]$DsRoot = "../SEED",
    [string]$OutputRoot = "cache/ds009999",
    [string]$LabelsMat = "",
    [string[]]$Subjects = @(),
    [string[]]$Sessions = @(),
    [ValidateSet("none", "subject", "loso")]
    [string]$SplitMode = "loso",
    [int]$TrainSubjects = 12,
    [int]$ValSubjects = 2,
    [int]$TestSubjects = 1,
    [double]$InputSfreq = 200.0,
    [double]$EegTargetSfreq = 200.0,
    [double]$WindowSec = 8.0,
    [double]$WindowOverlapSec = 0.0,
    [int]$EegSeqLen = 8,
    [int]$EegPatchLen = 200,
    [switch]$TrainingReady,
    [string]$TargetChannelManifest = "",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))

if (-not $PSBoundParameters.ContainsKey("TrainingReady")) {
    $TrainingReady = $true
}

$python = if ($PythonExe.Trim()) { $PythonExe } else { "python" }

$cliArgs = @(
    "preprocess/prepare_ds009999.py",
    "--ds-root", $DsRoot,
    "--output-root", $OutputRoot,
    "--pack-subject-files",
    "--split-mode", $SplitMode,
    "--train-subjects", $TrainSubjects.ToString(),
    "--val-subjects", $ValSubjects.ToString(),
    "--test-subjects", $TestSubjects.ToString(),
    "--input-sfreq", $InputSfreq.ToString(),
    "--eeg-target-sfreq", $EegTargetSfreq.ToString(),
    "--window-sec", $WindowSec.ToString(),
    "--window-overlap-sec", $WindowOverlapSec.ToString(),
    "--eeg-mode", "patched",
    "--eeg-seq-len", $EegSeqLen.ToString(),
    "--eeg-patch-len", $EegPatchLen.ToString()
)

if ($LabelsMat.Trim()) {
    $cliArgs += @("--labels-mat", $LabelsMat)
}
if ($TargetChannelManifest.Trim()) {
    $cliArgs += @("--target-channel-manifest", $TargetChannelManifest)
}
if ($TrainingReady) {
    $cliArgs += "--training-ready"
}
else {
    $cliArgs += "--no-training-ready"
}
if ($Subjects.Count -gt 0) {
    $cliArgs += "--subjects"
    $cliArgs += $Subjects
}
if ($Sessions.Count -gt 0) {
    $cliArgs += "--sessions"
    $cliArgs += $Sessions
}

Write-Host "Preparing ds009999 (SEED) dataset..."
Write-Host ("Output root: " + $OutputRoot)
Write-Host ("Split mode: " + $SplitMode)

& $python @cliArgs
if ($LASTEXITCODE -ne 0) {
    throw "ds009999 preprocessing failed with exit code $LASTEXITCODE"
}
