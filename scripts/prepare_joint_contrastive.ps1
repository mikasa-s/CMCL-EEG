param(
    [string]$Ds002336Root = "../ds002336",
    [string]$Ds002338Root = "../ds002338",
    [string]$Ds002739Root = "../ds002739",
    [string]$OutputRoot = "cache/joint_contrastive",
    [string[]]$Datasets = @("ds002336", "ds002739", "ds002338"),
    [string[]]$Ds002336Subjects = @(),
    [string[]]$Ds002338Subjects = @(),
    [string[]]$Ds002739Subjects = @(),
    [string[]]$Ds002336Tasks = @("motorloc", "MIpre", "MIpost", "eegNF", "fmriNF", "eegfmriNF"),
    [string[]]$Ds002338Tasks = @("MIpre", "MIpost", "1dNF", "2dNF"),
    [string[]]$Ds002739Runs = @(),
    [double]$EegWindowSec = 8.0,
    [bool]$TrainingReady = $true,
    [bool]$SkipExistingDatasets = $true,
    [int]$NumWorkers = 2,
    [string[]]$ForceRefreshDatasets = @(),
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
if ($env:CONDA_PREFIX) {
    $python = Join-Path $env:CONDA_PREFIX "python.exe"
}
else {
    $python = "python"
}
if ($PythonExe.Trim()) {
    $python = $PythonExe
}

$cliArgs = @(
    (Join-Path $repoRoot "preprocess/prepare_joint_contrastive.py"),
    "--output-root", $OutputRoot,
    "--eeg-window-sec", $EegWindowSec.ToString(),
    "--fmri-mode", "volume",
    "--fmri-voxel-size", "2.0", "2.0", "2.0",
    "--fmri-max-shape", "48", "48", "48",
    "--tr", "2.0",
    "--eeg-mode", "patched",
    "--eeg-target-sfreq", "200",
    "--eeg-lfreq", "0.5",
    "--eeg-hfreq", "40",
    "--num-workers", $NumWorkers.ToString(),
    "--pack-subject-files"
)

if ($Datasets.Count -gt 0) {
    $cliArgs += "--datasets"
    $cliArgs += $Datasets
}

if ($Datasets -contains "ds002336") {
    $cliArgs += @("--ds002336-root", $Ds002336Root)
    if ($Ds002336Subjects.Count -gt 0) {
        $cliArgs += "--ds002336-subjects"
        $cliArgs += $Ds002336Subjects
    }
    if ($Ds002336Tasks.Count -gt 0) {
        $cliArgs += "--ds002336-tasks"
        $cliArgs += $Ds002336Tasks
    }
    $cliArgs += "--ds002336-drop-ecg"
}

if ($Datasets -contains "ds002338") {
    $cliArgs += @("--ds002338-root", $Ds002338Root)
    if ($Ds002338Subjects.Count -gt 0) {
        $cliArgs += "--ds002338-subjects"
        $cliArgs += $Ds002338Subjects
    }
    if ($Ds002338Tasks.Count -gt 0) {
        $cliArgs += "--ds002338-tasks"
        $cliArgs += $Ds002338Tasks
    }
    $cliArgs += "--ds002338-drop-ecg"
}

if ($Datasets -contains "ds002739") {
    $cliArgs += @("--ds002739-root", $Ds002739Root)
    if ($Ds002739Subjects.Count -gt 0) {
        $cliArgs += "--ds002739-subjects"
        $cliArgs += $Ds002739Subjects
    }
    if ($Ds002739Runs.Count -gt 0) {
        $cliArgs += "--ds002739-runs"
        $cliArgs += $Ds002739Runs
    }
}

if ($TrainingReady) {
    $cliArgs += "--training-ready"
}
else {
    $cliArgs += "--no-training-ready"
}

if ($SkipExistingDatasets) {
    $cliArgs += "--skip-existing-datasets"
}

if ($ForceRefreshDatasets.Count -gt 0) {
    $cliArgs += "--force-refresh-datasets"
    $cliArgs += $ForceRefreshDatasets
}

Write-Host "Preparing joint contrastive cache..."
Write-Host ("Datasets: " + ($Datasets -join ", "))
Write-Host ("Output root: " + $OutputRoot)
Write-Host ("EEG window sec: " + $EegWindowSec)

& $python @cliArgs
if ($LASTEXITCODE -ne 0) {
    throw "joint dataset preprocessing failed with exit code $LASTEXITCODE"
}