param(
    [switch]$OpenViewer
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python virtual environment not found. Run: powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1"
}

& $Python -c "import mujoco; print('MuJoCo version:', mujoco.__version__); print('Package:', mujoco.__file__)"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$OfficialDir = Join-Path $ProjectRoot "artifacts\official_models"
$OfficialModel = Join-Path $OfficialDir "humanoid.xml"
$OfficialModelUrl = "https://raw.githubusercontent.com/google-deepmind/mujoco/main/model/humanoid/humanoid.xml"

if (-not (Test-Path -LiteralPath $OfficialModel)) {
    New-Item -ItemType Directory -Force -Path $OfficialDir | Out-Null
    Invoke-WebRequest -UseBasicParsing -Uri $OfficialModelUrl -OutFile $OfficialModel
}

$env:MUJOCO_VERIFY_XML = $OfficialModel
& $Python -c "import os, mujoco; path=os.environ['MUJOCO_VERIFY_XML']; model=mujoco.MjModel.from_xml_path(path); data=mujoco.MjData(model); mujoco.mj_step(model, data, nstep=100); print(f'official humanoid ok: nq={model.nq}, nv={model.nv}, nu={model.nu}, time={data.time:.4f}')"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($OpenViewer) {
    & $Python -m mujoco.viewer --mjcf=$OfficialModel
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
