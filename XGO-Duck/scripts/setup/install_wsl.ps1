$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $projectRoot
$Host.UI.RawUI.WindowTitle = 'XGO-Duck | WSL installation (visible, logged)'
$recordDir = Join-Path $projectRoot 'records/deployment/2026-10-31'
New-Item -ItemType Directory -Force -Path $recordDir | Out-Null
Start-Transcript -Path (Join-Path $recordDir 'wsl-install-transcript.txt') -Append
try {
    Write-Host 'Installing official Microsoft WSL. No automatic reboot is requested.'
    Write-Host 'COMMAND: wsl.exe --install --no-distribution --web-download'
    & wsl.exe --install --no-distribution --web-download 2>&1 | Tee-Object -FilePath (Join-Path $recordDir 'wsl-install-output.txt')
    $installExit = $LASTEXITCODE
    @{ finished_at = (Get-Date -Format o); command = 'wsl.exe --install --no-distribution --web-download'; exit_code = $installExit } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $recordDir 'wsl-install-result.json') -Encoding utf8
    Write-Host "Installer exit code: $installExit. Read output before proceeding; a reboot may be required."
} finally {
    Stop-Transcript
}
