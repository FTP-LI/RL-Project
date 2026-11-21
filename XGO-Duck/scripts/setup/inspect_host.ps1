$ErrorActionPreference = 'Continue'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $projectRoot
$Host.UI.RawUI.WindowTitle = 'XGO-Duck | Host environment inspection'
$recordDir = Join-Path $projectRoot 'records/deployment/2026-10-31'
New-Item -ItemType Directory -Force -Path $recordDir | Out-Null
Start-Transcript -Path (Join-Path $recordDir 'host-inspection.txt') -Append
Write-Host 'Read-only inspection. No packages are installed by this script.'
Get-Date -Format o
Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,TotalVisibleMemorySize,FreePhysicalMemory | Format-List
Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | Format-List
Get-PSDrive -PSProvider FileSystem | Select-Object Name,Used,Free | Format-Table
Write-Host 'COMMAND: nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | Out-Host
Write-Host 'COMMAND: available tools'
Get-Command wsl,git,python,uv,pwsh,arduino-cli -ErrorAction SilentlyContinue | Select-Object Name,Source | Format-Table
$wslService = Get-Service -Name WslService -ErrorAction SilentlyContinue
if ($wslService) {
    Write-Host 'COMMAND: wsl --list --verbose'
    wsl --list --verbose | Out-Host
    $distributions = @(Get-ChildItem 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss' -ErrorAction SilentlyContinue | ForEach-Object { (Get-ItemProperty $_.PSPath).DistributionName })
    if ('Ubuntu-24.04' -in $distributions) {
        Write-Host 'COMMAND: WSL Ubuntu-24.04 read-only probe'
        wsl -d Ubuntu-24.04 -- bash -lc 'cat /etc/os-release; command -v uv python3 git; python3 --version; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; df -h ~; free -h' | Out-Host
    } else {
        Write-Host 'Ubuntu-24.04 is not registered; Linux GPU inspection remains pending.'
    }
} else {
    Write-Host 'Modern WSL service is absent. Skip the Windows WSL stub to avoid its interactive installation prompt.'
}
Write-Host 'Inspection finished. This window remains open for review.'
Stop-Transcript
