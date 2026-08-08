param(
    [string]$Venv = ".venv-win",
    [string]$ReferenceVenv = "..\keqing1\.venv-win"
)

$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Repo

function Invoke-UvPip {
    param([string[]]$Packages)
    & uv pip install --python (Join-Path $Repo "$Venv\Scripts\python.exe") @Packages
    if ($LASTEXITCODE -ne 0) { throw "uv pip install failed" }
}

# 1. venv
if (-not (Test-Path (Join-Path $Repo "$Venv\Scripts\python.exe"))) {
    uv venv --python 3.12 $Venv
    if ($LASTEXITCODE -ne 0) { throw "uv venv failed" }
}

# 2. Python dependencies (uv cache makes this fast after the first run)
Invoke-UvPip @("numpy>=1.24", "riichienv==0.4.8", "torch>=2.11.0", "tensorboard>=2.20.0", "pytest>=9.0.2", "ruff>=0.15.10")

# 3. keqing_core wheel (requires cargo + maturin/uvx on PATH)
$Wheel = Join-Path $Repo "rust\keqing_core\target\wheels\keqing_core-0.1.0-cp312-cp312-win_amd64.whl"
if (-not (Test-Path $Wheel)) {
    & (Join-Path $Repo ".venv-win\Scripts\python.exe") (Join-Path $Repo "rust\keqing_core\build.py")
    if ($LASTEXITCODE -ne 0) { throw "keqing_core wheel build failed" }
}
Invoke-UvPip @($Wheel)

# 4. libriichi runtime bits (compiled extension + python package).  The python
#    package source is not vendored in this tree; copy it from a reference
#    environment (default: the keqing1 workspace venv) when missing.
$Site = Join-Path $Repo "$Venv\Lib\site-packages"
if (-not (Test-Path (Join-Path $Site "libriichi\__init__.py"))) {
    $RefSite = Join-Path (Resolve-Path $ReferenceVenv -ErrorAction SilentlyContinue) "Lib\site-packages"
    if (-not (Test-Path (Join-Path $RefSite "libriichi\__init__.py"))) {
        throw "libriichi python package not found under $RefSite; pass -ReferenceVenv"
    }
    Copy-Item -Recurse -LiteralPath (Join-Path $RefSite "libriichi") -Destination (Join-Path $Site "libriichi")
    if (-not (Test-Path (Join-Path $Site "riichi.pyd"))) {
        Copy-Item -LiteralPath (Join-Path $RefSite "riichi.pyd") -Destination (Join-Path $Site "riichi.pyd")
    }
}

"setup complete: $Venv"
