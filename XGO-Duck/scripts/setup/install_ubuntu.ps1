param([switch]$UseLocalPackage)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $projectRoot
$Host.UI.RawUI.WindowTitle = 'XGO-Duck | Ubuntu installation'
$recordDir = Join-Path $projectRoot ('records/deployment/' + (Get-Date -Format 'yyyy-MM-dd'))
New-Item -ItemType Directory -Force -Path $recordDir | Out-Null
$runId = Get-Date -Format 'HHmmss'
$resultPath = Join-Path $recordDir "ubuntu-install-$runId-result.json"
$result = [ordered]@{ started_at = (Get-Date -Format o); status = 'running'; distribution = 'Ubuntu-24.04' }
$scriptExit = 0
Start-Transcript -Path (Join-Path $recordDir "ubuntu-install-$runId-transcript.txt")
try {
    $checkpointPath = Join-Path $projectRoot 'records/deployment/2026-10-31/restart-checkpoint.json'
    $checkpoint = Get-Content -LiteralPath $checkpointPath -Raw | ConvertFrom-Json
    $boot = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime
    if (-not $boot) { throw 'Cannot verify Windows boot time.' }
    $result.boot_time = $boot.ToString('o')
    if ($checkpoint.reboot_required -and $boot -le [datetime]$checkpoint.boot_time_before_restart) {
        throw 'Windows restart is still required after enabling VirtualMachinePlatform. Save your work and restart manually.'
    }
    foreach ($probe in @('--version', '--status')) {
        Write-Host "COMMAND: wsl.exe $probe"
        & wsl.exe $probe
        $result[$probe.TrimStart('-') + '_exit_code'] = $LASTEXITCODE
        if ($LASTEXITCODE -ne 0) { throw "WSL probe $probe failed: exit $LASTEXITCODE" }
    }
    $registered = @(Get-ChildItem 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss' -ErrorAction SilentlyContinue |
        Get-ItemProperty | Where-Object DistributionName -eq 'Ubuntu-24.04')
    if ($registered.Count -eq 0) {
        if ($UseLocalPackage) {
            $downloadDir = Join-Path $projectRoot 'workspace/downloads'
            New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null
            $packagePath = Join-Path $downloadDir 'ubuntu-24.04.5-wsl-amd64.wsl'
            $packageUrl = 'https://releases.ubuntu.com/24.04.5/ubuntu-24.04.5-wsl-amd64.wsl'
            $expectedHash = 'bb415d824822c4b878125729af451a5d18fb13d1cf5cbed9a7393ad64ac6039e'
            $result.manifest_source = 'https://raw.githubusercontent.com/microsoft/WSL/master/distributions/DistributionInfo.json'
            $result.package_url = $packageUrl
            $result.expected_sha256 = $expectedHash
            $result.stage = 'downloading-local-package'
            $result | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $resultPath -Encoding utf8
            $validPackage = (Test-Path -LiteralPath $packagePath) -and ((Get-FileHash -LiteralPath $packagePath -Algorithm SHA256).Hash -eq $expectedHash)
            if (-not $validPackage) {
                Write-Host "COMMAND: curl.exe --fail --location --continue-at - --retry 3 --retry-all-errors --connect-timeout 20 --max-time 7200 --output $packagePath $packageUrl"
                & curl.exe --fail --location --continue-at - --retry 3 --retry-all-errors --connect-timeout 20 --max-time 7200 --output $packagePath $packageUrl 2>&1 | Tee-Object -FilePath (Join-Path $recordDir "ubuntu-download-$runId-output.txt")
                if ($LASTEXITCODE -ne 0) { throw "Ubuntu package download failed: curl exit $LASTEXITCODE" }
            }
            $result.actual_sha256 = (Get-FileHash -LiteralPath $packagePath -Algorithm SHA256).Hash.ToLowerInvariant()
            $result.package_bytes = (Get-Item -LiteralPath $packagePath).Length
            if ($result.actual_sha256 -ne $expectedHash -or $result.package_bytes -ne 388975696) {
                throw 'Ubuntu package SHA256 or size differs from the verified official release.'
            }
            $result.stage = 'installing-verified-package'
            $result | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $resultPath -Encoding utf8
            Write-Host "COMMAND: wsl.exe --install --from-file $packagePath --name Ubuntu-24.04 --version 2 --no-launch"
            & wsl.exe --install --from-file $packagePath --name Ubuntu-24.04 --version 2 --no-launch
        } else {
            Write-Host 'COMMAND: wsl.exe --install -d Ubuntu-24.04 --web-download --no-launch'
            & wsl.exe --install -d Ubuntu-24.04 --web-download --no-launch
        }
        $result.install_exit_code = $LASTEXITCODE
        if ($LASTEXITCODE -ne 0) { throw "Ubuntu installation failed: exit $LASTEXITCODE" }
        $result.status = 'install-command-succeeded-awaiting-initialization'
    } else {
        $result.status = 'existing-distribution-preserved'
        Write-Host 'Ubuntu-24.04 is already registered; preserving the existing distribution.'
    }
    Write-Host 'COMMAND: wsl.exe --list --verbose'
    & wsl.exe --list --verbose
    $result.list_exit_code = $LASTEXITCODE
    if ($LASTEXITCODE -ne 0) { throw "Distribution listing failed: exit $LASTEXITCODE" }
    Write-Host 'Installation log ends here. Initialize with: wsl.exe -d Ubuntu-24.04'
    Write-Host 'Enter your Linux username and password yourself, outside the transcript. GPU and training checks are still pending.'
} catch {
    $result.status = 'failed'
    $result.error = $_.Exception.Message
    $scriptExit = 1
    Write-Host $result.error -ForegroundColor Red
} finally {
    $result.finished_at = Get-Date -Format o
    $result.script_exit_code = $scriptExit
    $result | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $resultPath -Encoding utf8
    Stop-Transcript
}
exit $scriptExit
