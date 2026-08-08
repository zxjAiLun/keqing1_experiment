param(
    [string]$Venv = ".venv-win"
)

$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Repo

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
