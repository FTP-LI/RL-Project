# Sync local MuJoCo-Lab to Raspberry Pi.
# Edit these values when your Raspberry Pi address, username, or target path changes.

$PiHostName = "192.168.8.202"
$PiUser = "ftp"
$RemoteDir = "/home/ftp/MuJoCo-Lab"
$IdentityFile = "$env:USERPROFILE\.ssh\mujoco_lab_ed25519"

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $Name"
    }
}

function Quote-Remote {
    param([string]$Value)
    return "'" + $Value.Replace("'", "'\''") + "'"
}

function Ensure-SshKey {
    param(
        [string]$PrivateKeyPath,
        [string]$Comment
    )

    $keyDir = Split-Path -Parent $PrivateKeyPath
    $publicKeyPath = "$PrivateKeyPath.pub"

    New-Item -ItemType Directory -Force -Path $keyDir | Out-Null

    if (-not (Test-Path -LiteralPath $PrivateKeyPath)) {
        Write-Host "Creating SSH key: $PrivateKeyPath"
        & ssh-keygen -t ed25519 -f $PrivateKeyPath -N '""' -C $Comment
        if ($LASTEXITCODE -ne 0) {
            throw "ssh-keygen failed."
        }
    }

    if ((Test-Path -LiteralPath $PrivateKeyPath) -and -not (Test-Path -LiteralPath $publicKeyPath)) {
        Write-Host "Recreating missing public key: $publicKeyPath"
        & ssh-keygen -y -f $PrivateKeyPath | Set-Content -Path $publicKeyPath -Encoding ascii
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to recreate public key."
        }
    }

    if (-not (Test-Path -LiteralPath $publicKeyPath)) {
        throw "Public key not found: $publicKeyPath"
    }

    return $publicKeyPath
}

function Test-KeyLogin {
    param(
        [string[]]$SshOptions,
        [string]$RemoteTarget
    )

    & ssh @SshOptions $RemoteTarget "echo ssh-key-ok" 1>$null 2>$null
    return $LASTEXITCODE -eq 0
}

function Install-PublicKey {
    param(
        [string]$PublicKeyPath,
        [string]$RemoteTarget
    )

    $publicKey = (Get-Content -LiteralPath $PublicKeyPath -Raw).Trim()
    $escapedPublicKey = $publicKey.Replace("'", "'\''")
    $remoteCommand = "mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && grep -qxF '$escapedPublicKey' ~/.ssh/authorized_keys || echo '$escapedPublicKey' >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys"

    Write-Host "SSH key is not installed on this Raspberry Pi yet."
    Write-Host "Enter the Raspberry Pi password once. Future syncs should not ask again."
    & ssh -o StrictHostKeyChecking=accept-new $RemoteTarget $remoteCommand
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install SSH public key on Raspberry Pi."
    }
}

Require-Command "tar"
Require-Command "scp"
Require-Command "ssh"
Require-Command "ssh-keygen"

$WorkspaceRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$SourceDir = Join-Path $WorkspaceRoot "MuJoCo-Lab"

if (-not (Test-Path -LiteralPath $SourceDir)) {
    throw "Source directory does not exist: $SourceDir"
}

$SourceDir = (Resolve-Path $SourceDir).Path
$IdentityFile = $IdentityFile.Replace("~", $env:USERPROFILE)
$RemoteTarget = "${PiUser}@${PiHostName}"
$RemoteDirQuoted = Quote-Remote $RemoteDir
$ArchiveName = "mujoco-lab-sync.tar.gz"
$TempRoot = Join-Path $env:TEMP "mujoco-lab-sync"
$ArchivePath = Join-Path $TempRoot $ArchiveName
$RemoteArchive = "/tmp/$ArchiveName"
$RemoteArchiveQuoted = Quote-Remote $RemoteArchive

$PublicKeyPath = Ensure-SshKey -PrivateKeyPath $IdentityFile -Comment "mujoco-lab-$PiUser@$PiHostName"
$SshOptions = @(
    "-i", $IdentityFile,
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    "-o", "StrictHostKeyChecking=accept-new"
)

if (-not (Test-KeyLogin -SshOptions $SshOptions -RemoteTarget $RemoteTarget)) {
    Install-PublicKey -PublicKeyPath $PublicKeyPath -RemoteTarget $RemoteTarget

    if (-not (Test-KeyLogin -SshOptions $SshOptions -RemoteTarget $RemoteTarget)) {
        throw "SSH key login still failed after installing the public key."
    }
}

New-Item -ItemType Directory -Force -Path $TempRoot | Out-Null
if (Test-Path -LiteralPath $ArchivePath) {
    Remove-Item -LiteralPath $ArchivePath -Force
}

$ExcludeArgs = @(
    "--exclude=.git",
    "--exclude=.venv",
    "--exclude=__pycache__",
    "--exclude=runs",
    "--exclude=artifacts",
    "--exclude=logs",
    "--exclude=.pytest_cache",
    "--exclude=.mypy_cache",
    "--exclude=.ruff_cache",
    "--exclude=*.pyc",
    "--exclude=.DS_Store"
)

Write-Host "Source: $SourceDir"
Write-Host "Remote: ${RemoteTarget}:$RemoteDir"
Write-Host "Packing MuJoCo-Lab..."

Push-Location $SourceDir
try {
    & tar -czf $ArchivePath @ExcludeArgs .
    if ($LASTEXITCODE -ne 0) {
        throw "tar failed."
    }
}
finally {
    Pop-Location
}

Write-Host "Uploading archive..."
& ssh @SshOptions $RemoteTarget "mkdir -p $RemoteDirQuoted"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create remote directory."
}

& scp @SshOptions $ArchivePath "${RemoteTarget}:$RemoteArchive"
if ($LASTEXITCODE -ne 0) {
    throw "scp upload failed."
}

Write-Host "Extracting on Raspberry Pi..."
& ssh @SshOptions $RemoteTarget "mkdir -p $RemoteDirQuoted && tar -xzf $RemoteArchiveQuoted -C $RemoteDirQuoted && rm -f $RemoteArchiveQuoted"
if ($LASTEXITCODE -ne 0) {
    throw "Remote extraction failed."
}

Write-Host "Sync complete."

