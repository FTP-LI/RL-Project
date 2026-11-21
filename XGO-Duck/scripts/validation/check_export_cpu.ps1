param([Parameter(Mandatory)][ValidatePattern('^(baseline|candidate)_smoke_[0-9T]+$')][string]$RunId)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $projectRoot
$artifact = Join-Path $projectRoot "artifacts/$RunId"
$record = Join-Path $projectRoot "records/experiments/$RunId"
$reportPath = Join-Path $record 'onnx-runtime-1.22.1.json'
$python = Join-Path $projectRoot 'workspace/onnx-runtime/Scripts/python.exe'
& $python (Join-Path $PSScriptRoot 'check_onnx.py') --models (Join-Path $artifact 'policy.onnx') --report $reportPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
if (-not $report.passed -or $report.packages.onnxruntime -ne '1.22.1') { throw 'Expected a passing ONNX Runtime 1.22.1 report.' }
foreach ($path in @((Join-Path $artifact 'manifest.json'), (Join-Path $record 'artifact-manifest.json'))) {
    $manifest = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json -AsHashtable
    if ($manifest.onnx_sha256 -ne $report.models[0].sha256) { throw 'Report model hash differs from artifact manifest.' }
    $manifest.runtime_1_22_1_compatibility_verified = $true
    $manifest.runtime_compatibility_report = $reportPath
    $manifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $path -Encoding utf8
}
Write-Host 'PC ONNX Runtime 1.22.1 check passed; hardware remains untested.'
exit 0
