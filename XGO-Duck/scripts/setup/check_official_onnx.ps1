$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $projectRoot
$Host.UI.RawUI.WindowTitle = 'XGO-Duck | Official ONNX CPU verification'
$recordDir = Join-Path $projectRoot 'records/deployment/2026-10-31'
New-Item -ItemType Directory -Force -Path $recordDir | Out-Null
Start-Transcript -Path (Join-Path $recordDir 'onnx-check-transcript.txt') -Append
$result = [ordered]@{ started_at=(Get-Date -Format o); status='running'; stage='environment' }
try {
    $python = Join-Path $projectRoot 'workspace/onnx-runtime/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $python)) {
        Write-Host 'COMMAND: python -m venv workspace/onnx-runtime'
        & python -m venv workspace/onnx-runtime
        if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
    }
    Write-Host 'COMMAND: <isolated-python> -m pip install -r configs/environments/onnx_cpu_requirements.txt'
    & $python -m pip install --disable-pip-version-check -r configs/environments/onnx_cpu_requirements.txt 2>&1 | Tee-Object -FilePath (Join-Path $recordDir 'onnx-install-output.txt')
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed: $LASTEXITCODE" }
    & $python -m pip freeze | Set-Content -LiteralPath (Join-Path $recordDir 'onnx-packages.txt') -Encoding utf8
    $result.stage = 'inference'
    Write-Host 'COMMAND: check_onnx.py --models official_resources/xgoduck_runtime_arduino/python --report records/deployment/2026-10-31/official-onnx-cpu.json'
    & $python scripts/validation/check_onnx.py --models official_resources/xgoduck_runtime_arduino/python --report records/deployment/2026-10-31/official-onnx-cpu.json 2>&1 | Tee-Object -FilePath (Join-Path $recordDir 'onnx-check-output.txt')
    $result.check_exit_code = $LASTEXITCODE
    if ($LASTEXITCODE -ne 0) { throw "ONNX check failed: $LASTEXITCODE; inspect the JSON report" }
    $result.status = 'passed'
    Write-Host 'CPU interface checks finished. This is not simulation or robot validation.'
} catch {
    $result.status = 'failed'
    $result.error = $_.Exception.Message
    Write-Host "FAILED: $($_.Exception.Message)" -ForegroundColor Red
} finally {
    $result.finished_at = (Get-Date -Format o)
    $result | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $recordDir 'onnx-check-result.json') -Encoding utf8
    Stop-Transcript
}
