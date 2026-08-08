param(
    [switch]$PrepareOnly,
    [switch]$RunTraining
)

$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Repo
$env:UV_PROJECT_ENVIRONMENT = ".venv-win"

$ExpDir = "artifacts\experiments\model_pool_2026_07\V2_population_mixed_v4_warmstart_2026_07"
$LogDir = Join-Path $ExpDir "pipeline_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"

function Invoke-Logged {
    param([string]$Name, [string[]]$Command)
    $LogPath = Join-Path $LogDir "$RunStamp`_$Name.log"
    "[$(Get-Date -Format o)] START $Name" | Tee-Object -FilePath $LogPath -Append
    & $Command[0] @($Command[1..($Command.Count - 1)]) *>> $LogPath
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE. See $LogPath" }
    "[$(Get-Date -Format o)] DONE $Name" | Tee-Object -FilePath $LogPath -Append
}

Invoke-Logged "00_audit_v2_data" @(
    "uv", "run", "--no-sync", "python", "training/mortal/audit_population_synthetic_dataset.py"
)
Invoke-Logged "01_prepare_v2" @(
    "uv", "run", "--no-sync", "python", "training/mortal/prepare_v2_population_mixed_warmstart.py"
)

if ($PrepareOnly) { exit 0 }

if ($RunTraining) {
    $StatePath = Join-Path $ExpDir "mortal.pth"
    $ParentCheckpoint = "artifacts\mortal_training\checkpoints\mortal_default_70k_promoted_candidate.pth"
    $Train = @(
        "uv", "run", "--no-sync", "python", "training/run_mortal_dqn_offline.py",
        "--config", "$ExpDir\config.toml", "--target-steps", "74000",
        "--device", "cuda", "--num-workers", "0", "--seed", "20260712", "--data-seed", "20260712",
        "--archive-steps", "72000,74000", "--archive-dir", "$ExpDir\checkpoints", "--log-every", "50"
    )
    if (-not (Test-Path $StatePath)) {
        $Train += @(
            "--initialize-from", $ParentCheckpoint,
            "--initialize-optimizer-from", $ParentCheckpoint,
            "--initial-steps", "70000"
        )
    }
    Invoke-Logged "02_train_v2_74000" $Train
}
