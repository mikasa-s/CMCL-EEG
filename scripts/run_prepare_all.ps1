Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location "D:\OpenNeuro\EEG-fMRI-Contrastive"

Write-Host "Running ds002739 prepare script..."
& ".\scripts\ds002739\prepare_ds002739.ps1"

Write-Host "Running ds002336 prepare script..."
& ".\scripts\ds002336\prepare_ds002336.ps1"

Write-Host "All prepare scripts finished."