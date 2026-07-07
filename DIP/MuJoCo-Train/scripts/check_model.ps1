Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

& .\.venv\Scripts\python.exe -m pendulum_lab.scripts.check_model --config configs\double_rotary_pendulum.json
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
