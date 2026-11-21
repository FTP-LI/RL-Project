param([switch]$UseMirror)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$Host.UI.RawUI.WindowTitle = 'XGO-Duck | Linux training environment'
Set-Location -LiteralPath $projectRoot
$recordDir = Join-Path $projectRoot ('records/deployment/' + (Get-Date -Format 'yyyy-MM-dd'))
New-Item -ItemType Directory -Force -Path $recordDir | Out-Null
$runId = Get-Date -Format 'HHmmss'
$result = [ordered]@{ started_at = (Get-Date -Format o); status = 'running'; stage = 'path-conversion' }
$scriptExit = 0
Start-Transcript -Path (Join-Path $recordDir "linux-launch-$runId-transcript.txt")
try {
    $convertedPath = & wsl.exe -d Ubuntu-24.04 --exec wslpath -a (Join-Path $PSScriptRoot 'bootstrap_linux.sh')
    if ($LASTEXITCODE -ne 0 -or -not $convertedPath) { throw 'WSL path conversion failed; verify Ubuntu initialization and the script path.' }
    $linuxScript = ([string]$convertedPath).Trim()
    $result.linux_script = $linuxScript
    $result.stage = 'linux-bootstrap'
    Write-Host "Running foreground setup: $linuxScript"
    $linuxArgs = @('-d', 'Ubuntu-24.04', '--exec', 'bash', $linuxScript)
    if ($UseMirror) { $linuxArgs += '--mirror' }
    & wsl.exe @linuxArgs
    $result.bootstrap_exit_code = $LASTEXITCODE
    if ($LASTEXITCODE -ne 0) { throw "Linux bootstrap failed: exit $LASTEXITCODE" }
    $result.status = 'passed'
} catch {
    $result.status = 'failed'
    $result.error = $_.Exception.Message
    $scriptExit = 1
    Write-Host $result.error -ForegroundColor Red
} finally {
    $result.finished_at = Get-Date -Format o
    $result.script_exit_code = $scriptExit
    $result | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $recordDir "linux-launch-$runId-result.json") -Encoding utf8
    Stop-Transcript
}
exit $scriptExit
