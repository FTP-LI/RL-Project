param(
    [Parameter(Mandatory)] [ValidateSet('Archive', 'Restore')] [string] $Action,
    [Parameter(Mandatory)] [ValidatePattern('^[a-f0-9]{64}$')] [string] $ManifestSha256
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$records = Join-Path $root 'records'
$archive = Join-Path $records 'training-archive'
$areas = @('experiments', 'evaluations', 'benchmarks', 'profiling')
$reparse = [System.IO.FileAttributes]::ReparsePoint

function Assert-PhysicalDirectory([string] $Path) {
    $item = Get-Item -LiteralPath $Path -Force
    if (-not $item.PSIsContainer -or ($item.Attributes -band $reparse) -or
        (Resolve-Path -LiteralPath $Path).Path -cne $Path) {
        throw "Expected a physical directory at $Path"
    }
}

function Assert-ArchiveLink([string] $Path, [string] $Target) {
    $item = Get-Item -LiteralPath $Path -Force
    if ($item.LinkType -ne 'Junction' -or $item.Target -ine $Target) {
        throw "Unexpected compatibility link at $Path"
    }
}

function Move-TrainingArea([string] $Name, [string] $Direction) {
    $source = Join-Path $records $Name
    $target = Join-Path $archive $Name
    if ($Direction -eq 'Archive') {
        Assert-PhysicalDirectory $source
        if (Test-Path -LiteralPath $target) { throw "Archive target already exists at $target" }
        Move-Item -LiteralPath $source -Destination $target
        try {
            New-Item -ItemType Junction -Path $source -Target $target | Out-Null
        } catch {
            Move-Item -LiteralPath $target -Destination $source
            throw
        }
    } else {
        Assert-PhysicalDirectory $target
        Assert-ArchiveLink $source $target
        # Remove the verified junction itself, then restore the physical folder.
        [System.IO.Directory]::Delete($source)
        try {
            Move-Item -LiteralPath $target -Destination $source
        } catch {
            New-Item -ItemType Junction -Path $source -Target $target | Out-Null
            throw
        }
    }
}

Assert-PhysicalDirectory $records
Assert-PhysicalDirectory $archive
$manifest = Join-Path $archive 'manifest.json'
if ((Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash -ine $ManifestSha256) {
    throw 'The archive manifest has changed.'
}
$project = Get-Content -Raw -LiteralPath (Join-Path $root 'project.json') | ConvertFrom-Json
if ($project.goal_execution.status -ne 'paused') { throw 'Pause the staged training goal first.' }

# Verify all absolute sources and targets before any directory move.
foreach ($name in $areas) {
    $source = Join-Path $records $name
    $target = Join-Path $archive $name
    if ($Action -eq 'Archive') {
        Assert-PhysicalDirectory $source
        if (Test-Path -LiteralPath $target) { throw "Archive target already exists at $target" }
    } else {
        Assert-PhysicalDirectory $target
        Assert-ArchiveLink $source $target
    }
}

$done = [System.Collections.Generic.List[string]]::new()
try {
    foreach ($name in $areas) {
        Move-TrainingArea $name $Action
        $done.Add($name)
    }
} catch {
    $rollback = if ($Action -eq 'Archive') { 'Restore' } else { 'Archive' }
    $done.Reverse()
    foreach ($name in $done) { Move-TrainingArea $name $rollback }
    throw
}
[pscustomobject] @{ action = $Action; archive = $archive; areas = $done } | ConvertTo-Json
