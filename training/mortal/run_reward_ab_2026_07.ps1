param(
    [switch]$PrepareOnly,
    [switch]$RunTraining,
    [switch]$FirstPairOnly
)

$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Repo
$env:UV_PROJECT_ENVIRONMENT = ".venv-win"

$ExpDir = "artifacts\experiments\model_pool_2026_07\reward_ab_2026_07_epoch2"
$GrpCheckpoint = "artifacts\experiments\model_pool_2026_07\keqing_grp_v1\keqing_grp_v1_best.pth"
$Manifest = Join-Path $ExpDir "manifest.json"
$Preflight = Join-Path $ExpDir "grp_reward_preflight_full.json"
$LogDir = Join-Path $ExpDir "pipeline_logs"
$RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Seeds = @(20260718, 20260719, 20260720)
$Groups = @(
    @{ Id = "F_final_rank_mc_weights_only"; Mode = "final_rank_mc" },
    @{ Id = "G_mortal_grp_delta_pt_weights_only"; Mode = "mortal_grp_delta_pt" }
)

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Invoke-Logged {
    param([string]$Name, [string[]]$Command)
    $LogPath = Join-Path $LogDir "$RunStamp`_$Name.log"
    "[$(Get-Date -Format o)] START $Name" | Tee-Object -FilePath $LogPath -Append
    $PreviousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    & $Command[0] @($Command[1..($Command.Count - 1)]) 2>&1 | Tee-Object -FilePath $LogPath -Append
    $ExitCode = $LASTEXITCODE
    $ErrorActionPreference = $PreviousErrorAction
    $Error.Clear()
    if ($ExitCode -ne 0) { throw "$Name failed with exit code $ExitCode. See $LogPath" }
    "[$(Get-Date -Format o)] DONE $Name" | Tee-Object -FilePath $LogPath -Append
}

if (-not (Test-Path -LiteralPath $GrpCheckpoint)) {
    throw "Frozen GRP checkpoint is missing: $GrpCheckpoint"
}

Invoke-Logged "00_prepare_reward_ab" @(
    "uv", "run", "--no-sync", "python", "training/mortal/prepare_reward_ab.py",
    "--grp-checkpoint", $GrpCheckpoint
)

if (-not (Test-Path -LiteralPath $Preflight)) {
    Invoke-Logged "01_reward_preflight" @(
        "uv", "run", "--no-sync", "python", "training/mortal/preflight_reward_distribution.py",
        "--config", "$ExpDir\G_mortal_grp_delta_pt_weights_only\seed_20260718\config.toml",
        "--output", $Preflight
    )
}

if ($PrepareOnly -or -not $RunTraining) { exit 0 }

$PairSeeds = $Seeds
if ($FirstPairOnly) { $PairSeeds = @($Seeds[0]) }
foreach ($seed in $PairSeeds) {
    foreach ($group in $Groups) {
        $RunDir = Join-Path $ExpDir "$($group.Id)\seed_$seed"
        $StatePath = Join-Path $RunDir "mortal.pth"
        $Command = @(
            "uv", "run", "--no-sync", "python", "training/run_mortal_dqn_offline.py",
            "--config", "$RunDir\config.toml", "--target-steps", "72000",
            "--device", "cuda", "--num-workers", "0", "--seed", "$seed", "--data-seed", "$seed",
            "--archive-steps", "72000", "--archive-dir", "$RunDir\checkpoints", "--log-every", "50"
        )
        if (Test-Path -LiteralPath $StatePath) {
            $Command += @("--allow-legacy-data-replay")
        } else {
            $Command += @(
                "--initialize-from", "artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth",
                "--initial-steps", "70000"
            )
        }
        Invoke-Logged "$($group.Id)_$seed" $Command
    }
}
