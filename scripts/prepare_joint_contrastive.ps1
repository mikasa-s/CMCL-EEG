param(
    [string]$Ds002336Root = "../ds002336",
    [string]$Ds002739Root = "../ds002739",
    [string]$OutputRoot = "cache/joint_contrastive",
    [string[]]$Datasets = @("ds002336", "ds002739"),
    [string[]]$Ds002336Subjects = @(),
    [string[]]$Ds002739Subjects = @(),
    [string[]]$Ds002336Tasks = @("motorloc", "MIpre", "MIpost", "eegNF", "fmriNF", "eegfmriNF"),
    [string[]]$Ds002739Runs = @(),
    [double]$EegWindowSec = 8.0,
    [bool]$TrainingReady = $true
)

$ErrorActionPreference = "Stop"

Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))

$python = "python"

$cliArgs = @(
    "preprocess/prepare_joint_contrastive.py",
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

Write-Host "Preparing joint contrastive cache..."
Write-Host ("Datasets: " + ($Datasets -join ", "))
Write-Host ("Output root: " + $OutputRoot)
Write-Host ("EEG window sec: " + $EegWindowSec)

& $python @cliArgs