$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$Host.UI.RawUI.WindowTitle = 'XGO-Duck | WSL network recovery'
$recordDir = Join-Path $projectRoot ('records/deployment/' + (Get-Date -Format 'yyyy-MM-dd'))
New-Item -ItemType Directory -Force -Path $recordDir | Out-Null
$runId = Get-Date -Format HHmmss
$configPath = Join-Path ([Environment]::GetFolderPath('UserProfile')) '.wslconfig'
$result = [ordered]@{ started_at = (Get-Date -Format o); status = 'running'; config_path = $configPath; previous_config_existed = (Test-Path -LiteralPath $configPath) }
$scriptExit = 0
Start-Transcript -Path (Join-Path $recordDir "network-recovery-$runId-transcript.txt")
try {
    if ($result.previous_config_existed) { throw 'Existing .wslconfig found; inspect and merge explicitly instead of overwriting.' }
    Write-Host 'Stopping only the verified XGO-Duck uv sync before restarting WSL.'
    & wsl.exe -d Ubuntu-24.04 --exec bash -c 'if [[ -r /proc/432/cmdline ]] && [[ "$(readlink /proc/432/cwd)" == /home/atlas/XGO-Duck/workspace/xgoduck_rl ]] && [[ "$(cat /proc/432/comm)" == uv ]]; then kill -INT 432; for i in {1..20}; do kill -0 432 2>/dev/null || exit 0; sleep 1; done; exit 1; fi'
    if ($LASTEXITCODE -ne 0) { throw 'uv did not stop cleanly; WSL shutdown was not attempted.' }
    $content = "[wsl2]`nnetworkingMode=mirrored`nautoProxy=true`ndnsTunneling=true`n"
    [IO.File]::WriteAllText($configPath, $content, [Text.UTF8Encoding]::new($false))
    Copy-Item -LiteralPath $configPath -Destination (Join-Path $recordDir "network-recovery-$runId.wslconfig")
    $result.config_sha256 = (Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Host 'Restarting WSL only; Windows is not restarted. The idle Ubuntu shell will close.'
    & wsl.exe --shutdown
    if ($LASTEXITCODE -ne 0) { throw 'WSL shutdown failed.' }
    & wsl.exe -d Ubuntu-24.04 --exec wslinfo --networking-mode
    $result.network_mode_exit_code = $LASTEXITCODE
    Write-Host 'Testing the existing Windows localhost proxy from WSL with TLS verification enabled.'
    & wsl.exe -d Ubuntu-24.04 --exec curl --proxy http://127.0.0.1:7890 --fail --silent --show-error --location --connect-timeout 15 --max-time 45 https://pypi.org/simple/setuptools/ -o /dev/null -w 'http=%{http_code} bytes=%{size_download} seconds=%{time_total}'
    $result.pypi_proxy_exit_code = $LASTEXITCODE
    if ($LASTEXITCODE -ne 0) { throw 'WSL proxy path still fails; do not restart dependency installation yet.' }
    $result.status = 'passed'
} catch {
    $result.status = 'failed'
    $result.error = $_.Exception.Message
    $scriptExit = 1
    Write-Host $result.error -ForegroundColor Red
} finally {
    $result.finished_at = (Get-Date -Format o)
    $result | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $recordDir "network-recovery-$runId-result.json") -Encoding utf8
    Stop-Transcript
}
exit $scriptExit
