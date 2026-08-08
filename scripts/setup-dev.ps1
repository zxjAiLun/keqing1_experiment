param(
    [string]$Venv = ".venv-win",
    [switch]$SkipWheelPublish
)

$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Repo

$DataRoot = if ($env:KEQING_DATA_ROOT) { $env:KEQING_DATA_ROOT } else { Join-Path (Resolve-Path (Join-Path $Repo "..\..")) "keqing-data" }
$RuntimeWheelDir = Join-Path $DataRoot "runtime\keqing_core"
$VenvPython = Join-Path $Repo "$Venv\Scripts\python.exe"
$Site = Join-Path $Repo "$Venv\Lib\site-packages"

function Invoke-UvPip {
    param([string[]]$Packages)
    & uv pip install --python $VenvPython @Packages
    if ($LASTEXITCODE -ne 0) { throw "uv pip install failed" }
}

# 1. venv
if (-not (Test-Path $VenvPython)) {
    uv venv --python 3.12 $Venv
    if ($LASTEXITCODE -ne 0) { throw "uv venv failed" }
}

# 2. Python dependencies (uv cache makes this fast after the first run)
Invoke-UvPip @("mahjong>=1.4.0", "numpy>=1.24", "riichienv==0.4.8", "torch>=2.11.0", "tensorboard>=2.20.0", "pytest>=9.0.2", "ruff>=0.15.10")

# 3. libriichi runtime, built from the vendored Mortal crate
#    (riichi.pyd = third_party/Mortal/target/release/riichi.dll, package
#    surface = third_party/libriichi shims).  Requires cargo on PATH.
$env:PYO3_PYTHON = $VenvPython
# D3 exploration needs the DecisionContext extension; apply the patch once.
$NativeRoot = Join-Path $Repo "third_party\Mortal"
$D3Patch = Join-Path $Repo "training\mortal\patches\libriichi_d3_decision_context.patch"
$ContextSource = Join-Path $NativeRoot "libriichi\srcgent\defs.rs"
if ((Test-Path $ContextSource) -and (Test-Path $D3Patch)) {
    $hasDecisionContext = Select-String -LiteralPath $ContextSource -Pattern "pub struct DecisionContext" -Quiet
    if (-not $hasDecisionContext) {
        Write-Host "Applying D3 decision-context patch to the local Mortal checkout..."
        git -C $NativeRoot apply --whitespace=nowarn -- $D3Patch
        if ($LASTEXITCODE -ne 0) { throw "failed to apply D3 decision-context patch" }
    }
}
cargo build --manifest-path (Join-Path $Repo "third_party\Mortal\Cargo.toml") -p libriichi --lib --release
if ($LASTEXITCODE -ne 0) { throw "libriichi cargo build failed" }
Copy-Item -LiteralPath (Join-Path $Repo "third_party\Mortal\target\release\riichi.dll") -Destination (Join-Path $Site "riichi.pyd") -Force
if (Test-Path (Join-Path $Site "libriichi")) { Remove-Item -Recurse -Force (Join-Path $Site "libriichi") }
Copy-Item -Recurse -LiteralPath (Join-Path $Repo "third_party\libriichi\libriichi") -Destination (Join-Path $Site "libriichi")
& $VenvPython -c "from libriichi.arena import OneVsThree; assert hasattr(OneVsThree, 'py_selfplay'); print('libriichi OK')"
if ($LASTEXITCODE -ne 0) { throw "libriichi install verification failed" }

# 4. keqing_core wheel: build from rust/keqing_core, install, and publish a
#    copy to the shared data root for the Workbench repo to consume.
$Wheel = Join-Path $Repo "rust\keqing_core\target\wheels\keqing_core-0.1.0-cp312-cp312-win_amd64.whl"
if (-not (Test-Path $Wheel)) {
    & $VenvPython (Join-Path $Repo "rust\keqing_core\build.py")
    if ($LASTEXITCODE -ne 0) { throw "keqing_core wheel build failed" }
}
Invoke-UvPip @($Wheel)
if (-not $SkipWheelPublish) {
    New-Item -ItemType Directory -Force -Path $RuntimeWheelDir | Out-Null
    Copy-Item -LiteralPath $Wheel -Destination (Join-Path $RuntimeWheelDir (Split-Path $Wheel -Leaf)) -Force
    Write-Output "published keqing_core wheel to $RuntimeWheelDir"
}
& $VenvPython -c "import keqing_core; assert keqing_core.is_available(); print('keqing_core OK (rust available)')"
if ($LASTEXITCODE -ne 0) { throw "keqing_core verification failed" }

"setup complete: $Venv"
