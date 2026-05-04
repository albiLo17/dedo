# Run the three privileged conditions sequentially with wandb logging.
# Usage (from repo root, with the dedo conda env activated):
#   powershell -File experiments\hang_obs_exp\scripts\train_privileged_all.ps1
$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$LogRoot  = Join-Path $RepoRoot 'logs\hang_obs_exp'
$Python   = if ($env:PYTHON) { $env:PYTHON } else { 'python' }

Set-Location $RepoRoot

foreach ($cond in 'hole_centroid', 'hole_vertices', 'full_mesh') {
    $logdir  = Join-Path $LogRoot $cond
    $logfile = Join-Path $logdir 'train.log'
    New-Item -ItemType Directory -Force -Path $logdir | Out-Null

    Write-Host "=== Starting $cond -> $logfile ==="
    # Merge stdout+stderr inside cmd.exe so PowerShell 5.1 doesn't wrap stderr
    # lines as NativeCommandError ErrorRecords.
    $extra = if ($args) { ' ' + ($args -join ' ') } else { '' }
    $cmdLine = "`"$Python`" experiments\hang_obs_exp\scripts\train_privileged.py --obs_mode $cond --use_wandb$extra 2>&1"
    & cmd /c $cmdLine | Tee-Object -FilePath $logfile
    if ($LASTEXITCODE -ne 0) {
        throw "Condition $cond failed with exit code $LASTEXITCODE"
    }
    Write-Host "=== Finished $cond ==="
}

Write-Host "All privileged conditions done."
