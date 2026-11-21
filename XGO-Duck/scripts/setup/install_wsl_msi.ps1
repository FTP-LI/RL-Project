param([switch]$DownloadOnly)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $projectRoot
$Host.UI.RawUI.WindowTitle = 'XGO-Duck | Microsoft WSL installer'
$recordDir = Join-Path $projectRoot 'records/deployment/2026-10-31'
$downloadDir = Join-Path $projectRoot 'workspace/downloads'
New-Item -ItemType Directory -Force -Path $recordDir,$downloadDir | Out-Null
$resultPath = Join-Path $recordDir 'wsl-msi-result.json'
$result = [ordered]@{ started_at = (Get-Date -Format o); status = 'running'; stage = 'preflight'; reboot_required = $false }
Start-Transcript -Path (Join-Path $recordDir 'wsl-msi-transcript.txt') -Append
try {
    # Cancel only the exact earlier task installer, never arbitrary WSL sessions.
    $previous = Get-CimInstance Win32_Process -Filter 'ProcessId = 17096' -ErrorAction SilentlyContinue
    if ($previous -and $previous.Name -eq 'wsl.exe' -and $previous.CommandLine -like '*--install --no-distribution --web-download*') {
        Write-Host 'Stopping the earlier task installer because it produced no progress output.'
        Stop-Process -Id $previous.ProcessId -Force
    }
    $msi = Join-Path $downloadDir 'wsl.3.0.3.0.x64.msi'
    $url = 'https://github.com/microsoft/WSL/releases/download/3.0.3/wsl.3.0.3.0.x64.msi'
    $expectedHash = 'ae977286974255bd26bf925f00f40f5692c6b2084155cbc0563c09aea091f211'
    $result.stage = 'download'
    $result.url = $url
    $result.expected_sha256 = $expectedHash
    $result | ConvertTo-Json | Set-Content -LiteralPath $resultPath -Encoding utf8
    $validDownload = (Test-Path -LiteralPath $msi) -and ((Get-FileHash -LiteralPath $msi -Algorithm SHA256).Hash -eq $expectedHash)
    if (-not $validDownload) {
        Write-Host "Downloading official Microsoft WSL: $url (367575040 bytes)"
        Write-Host 'COMMAND: curl.exe --fail --location --continue-at - --retry 2 --retry-all-errors --connect-timeout 20 --max-time 3600 --output <project-download> <official-url>'
        & curl.exe --fail --location --continue-at - --retry 2 --retry-all-errors --connect-timeout 20 --max-time 3600 --output $msi $url 2>&1 | Tee-Object -FilePath (Join-Path $recordDir 'wsl-download-output.txt') -Append
        if ($LASTEXITCODE -ne 0) { throw "Download failed: curl exit $LASTEXITCODE" }
    }
    $actualHash = (Get-FileHash -LiteralPath $msi -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $expectedHash) { throw "SHA256 mismatch: $actualHash" }
    $signature = Get-AuthenticodeSignature -FilePath $msi
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notlike '*Microsoft Corporation*') {
        throw "Microsoft signature verification failed: $($signature.Status)"
    }
    $result.actual_sha256 = $actualHash
    $result.signature = [string]$signature.Status
    $result.signer = $signature.SignerCertificate.Subject
    if ($DownloadOnly) {
        $result.status = 'downloaded-and-verified'
        $result.stage = 'awaiting-installation'
        Write-Host 'Official MSI download, SHA256 and Microsoft signature verified. System installation has not run.'
        return
    }
    $result.stage = 'msi-install'
    $result | ConvertTo-Json | Set-Content -LiteralPath $resultPath -Encoding utf8
    Write-Host 'Hash and Microsoft signature verified. Starting visible Windows Installer; no automatic reboot.'
    $msiLog = Join-Path $recordDir 'wsl-msi.log'
    $install = Start-Process -FilePath msiexec.exe -WindowStyle Normal -Wait -PassThru -ArgumentList @('/i',('"' + $msi + '"'),'/passive','/norestart','/L*v',('"' + $msiLog + '"'))
    $result.msi_exit_code = $install.ExitCode
    if ($install.ExitCode -notin @(0,3010)) { throw "MSI failed: exit $($install.ExitCode)" }
    if ($install.ExitCode -eq 3010) { $result.reboot_required = $true }
    $result.stage = 'virtual-machine-platform'
    $feature = Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform
    $result.feature_before = [string]$feature.State
    if ($feature.State -ne 'Enabled') {
        Write-Host 'COMMAND: dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart'
        & dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart 2>&1 | Tee-Object -FilePath (Join-Path $recordDir 'virtual-machine-platform-output.txt')
        $result.feature_exit_code = $LASTEXITCODE
        if ($LASTEXITCODE -notin @(0,3010)) { throw "VirtualMachinePlatform failed: exit $LASTEXITCODE" }
        $result.reboot_required = $true
    }
    $result.feature_after = [string](Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform).State
    Write-Host 'COMMAND: wsl.exe --version'
    & wsl.exe --version 2>&1 | Tee-Object -FilePath (Join-Path $recordDir 'wsl-version-output.txt')
    $result.version_exit_code = $LASTEXITCODE
    if ($LASTEXITCODE -ne 0) { throw "WSL version probe failed: exit $LASTEXITCODE" }
    $result.status = 'installed'
    $result.stage = 'finished'
    Write-Host "WSL installation finished. Reboot required: $($result.reboot_required). Ubuntu and training packages still need setup."
} catch {
    $result.status = 'failed'
    $result.error = $_.Exception.Message
    Write-Host "FAILED: $($_.Exception.Message)" -ForegroundColor Red
} finally {
    $result.finished_at = (Get-Date -Format o)
    $result | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $resultPath -Encoding utf8
    Stop-Transcript
}
