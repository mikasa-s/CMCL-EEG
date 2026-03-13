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
    [string]$TargetChannelManifest = "",
    [ValidateSet("raw", "spm_unsmoothed", "spm_smoothed")]
    [string]$FmriSource = "spm_smoothed",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))

$python = if ($PythonExe.Trim()) { $PythonExe } else { "python" }

if (-not $Tasks -or $Tasks.Count -eq 0) {
    if ($DatasetName -eq "ds002336") {
        $Tasks = @("motorloc", "MIpre", "MIpost", "eegNF", "fmriNF", "eegfmriNF")
    }
    elseif ($DatasetName -eq "ds002338") {
        $Tasks = @("MIpre", "MIpost", "1dNF", "2dNF")
    }
}

# Auto-adjust split sizes by dataset when user does not explicitly override them.
if ($DatasetName -eq "ds002336") {
    $TrainSubjects = 7
    $ValSubjects = 2
    $TestSubjects = 1
}
elseif ($DatasetName -eq "ds002338") {
    $TrainSubjects = 14
    $ValSubjects = 2
    $TestSubjects = 1
}

if ($SplitMode -ne "none" -and $Subjects.Count -gt 0) {
    $requiredSubjects = if ($SplitMode -eq "loso") { $ValSubjects + 1 } else { $TrainSubjects + $ValSubjects + $TestSubjects }
    if ($Subjects.Count -lt $requiredSubjects) {
        Write-Warning "Provided subject subset is smaller than the requested split sizes; disabling split generation for this run."
        $SplitMode = "none"
    }
}

$cliArgs = @(
    "preprocess/prepare_ds00233x.py",
    "--ds-root", $DsRoot,
    "--output-root", $OutputRoot,
    "--sample-mode", "block",
    "--label-mode", "binary_rest_task",
    "--fmri-mode", "volume",
    "--pack-subject-files",
    "--eeg-mode", "patched",
    "--eeg-seq-len", $EegSeqLen.ToString(),
    "--eeg-patch-len", $EegPatchLen.ToString(),
    "--tr", "2.0",
    "--fmri-max-shape", "48", "48", "48",
    "--split-mode", $SplitMode,
    "--train-subjects", $TrainSubjects.ToString(),
    "--val-subjects", $ValSubjects.ToString(),
    "--test-subjects", $TestSubjects.ToString(),
    "--dataset-name", $DatasetName
)

if ($FmriSource -ne "raw") {
    $cliArgs += @(
        "--fmri-source", $FmriSource,
        "--discard-initial-trs", "0",
        "--protocol-offset-sec", "0.0"
    )
}
else {
    $discardInitialTrs = if ($DatasetName -eq "ds002338") { "2" } else { "1" }
    $cliArgs += @(
        "--discard-initial-trs", $discardInitialTrs,
        "--protocol-offset-sec", "2.0"
    )
}

$cliArgs += "--tasks"
$cliArgs += $Tasks

if ($DropEcg) {
    $cliArgs += "--drop-ecg"
}
if ($TrainingReady) {
    $cliArgs += "--training-ready"
}
else {
    $cliArgs += "--no-training-ready"
}
if ($EegOnly) {
    $cliArgs += "--eeg-only"
}
if ($TargetChannelManifest.Trim()) {
    $cliArgs += @("--target-channel-manifest", $TargetChannelManifest)
}
if ($Subjects.Count -gt 0) {
    $cliArgs += "--subjects"
    $cliArgs += $Subjects
}

Write-Host ("Preparing " + $DatasetName + " dataset...")
Write-Host ("Output root: " + $OutputRoot)
Write-Host ("Split mode: " + $SplitMode)
Write-Host ("fMRI source: " + $FmriSource)
Write-Host ("EEG-only: " + $EegOnly)

& $python @cliArgs
