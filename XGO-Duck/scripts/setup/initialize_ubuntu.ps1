$ErrorActionPreference = 'Stop'
$Host.UI.RawUI.WindowTitle = 'XGO-Duck | Ubuntu first start'
Set-Location -LiteralPath (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Write-Host 'Initialize your Linux username and password in this visible window.'
Write-Host 'No transcript is active here. Do not send your password in chat.'
& wsl.exe -d Ubuntu-24.04
