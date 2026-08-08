param(
    [string]$Venv = ".venv-win"
)

$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Repo

$NativeRoot = Join-Path $Repo "third_party\Mortal"
$D3Patch = Join-Path $Repo "training\mortal\patches\libriichi_d3_decision_context.patch"
$ContextSource = Join-Path $NativeRoot "libriichi\srcgent\defs.rs"
if ((Test-Path $ContextSource) -and (Test-Path $D3Patch)) {
    $hasDecisionContext = Select-String -LiteralPath $ContextSource -Pattern "pub struct DecisionContext" -Quiet
    if (-not $hasDecisionContext) {
        Write-Host "Applying D3 decision-context patch to the local Mortal checkout..."
        git -C $NativeRoot apply --whitespace=nowarn -- $D3Patch
        if ($LASTEXITCODE -ne 0) {
            throw "failed to apply D3 decision-context patch"
        }
    }
}

$Manifest = Join-Path $Repo "third_party\Mortal\Cargo.toml"
$SourceDll = Join-Path $Repo "third_party\Mortal\target\release\riichi.dll"
$TargetPyd = Join-Path $Repo "$Venv\Lib\site-packages\riichi.pyd"

cargo build --manifest-path $Manifest -p libriichi --lib --release
if ($LASTEXITCODE -ne 0) {
    throw "libriichi release build failed with exit code $LASTEXITCODE"
}
if (-not (Test-Path $TargetPyd)) {
    throw "target Python extension does not exist: $TargetPyd"
}

$Backup = "$TargetPyd.bak"
if (-not (Test-Path $Backup)) {
    Copy-Item -LiteralPath $TargetPyd -Destination $Backup
}
Copy-Item -LiteralPath $SourceDll -Destination $TargetPyd -Force

$env:UV_PROJECT_ENVIRONMENT = $Venv
uv run --no-sync python -c "from libriichi.arena import OneVsThree; assert hasattr(OneVsThree, 'py_selfplay'); print(OneVsThree.py_selfplay.__doc__)"
if ($LASTEXITCODE -ne 0) {
    throw "installed libriichi extension does not expose OneVsThree.py_selfplay"
}
