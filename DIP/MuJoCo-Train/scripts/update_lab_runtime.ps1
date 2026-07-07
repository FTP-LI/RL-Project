Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$TrainRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$LearnRoot = Resolve-Path (Join-Path $TrainRoot "..")
$LabRoot = Join-Path $LearnRoot "MuJoCo-Lab"

if (-not (Test-Path -LiteralPath $LabRoot)) {
    throw "MuJoCo-Lab not found: $LabRoot"
}

$ItemsToSync = @(
    "assets",
    "configs",
    "hardware",
    "pendulum_lab",
    "requirements.txt"
)

foreach ($item in $ItemsToSync) {
    $source = Join-Path $TrainRoot $item
    $target = Join-Path $LabRoot $item

    if (-not (Test-Path -LiteralPath $source)) {
        throw "Missing source item: $source"
    }

    if (Test-Path -LiteralPath $target) {
        Remove-Item -LiteralPath $target -Recurse -Force
    }

    Copy-Item -LiteralPath $source -Destination $target -Recurse
    Write-Host "Updated: $target"
}

Write-Host "MuJoCo-Lab runtime updated."

