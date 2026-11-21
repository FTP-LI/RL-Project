param([Parameter(Mandatory)][ValidateSet('baseline','candidate')][string]$Mode)
$ErrorActionPreference = 'Stop'
$Host.UI.RawUI.WindowTitle = "XGO-Duck | $Mode smoke training"
$convertedPath = & wsl.exe -d Ubuntu-24.04 --exec wslpath -a (Join-Path $PSScriptRoot 'run_smoke.sh')
if ($LASTEXITCODE -ne 0 -or -not $convertedPath) { throw 'Cannot resolve the Linux training script.' }
& wsl.exe -d Ubuntu-24.04 --exec bash ([string]$convertedPath).Trim() $Mode
$trainingExit = $LASTEXITCODE
Write-Host "Training wrapper exit: $trainingExit"
exit $trainingExit
