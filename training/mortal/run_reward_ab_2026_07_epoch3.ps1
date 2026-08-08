param(
    [switch]$PrepareOnly,
    [switch]$RunTraining,
    [switch]$FirstPairOnly
)

$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Repo
$Python = Join-Path $Repo ".venv-win\Scripts\python.exe"
$env:UV_PROJECT_ENVIRONMENT = ".venv-win"

$ExpId = "reward_ab_2026_07_epoch3"
$ExpDir = "artifacts\experiments\model_pool_2026_07\$ExpId"
$GrpCheckpoint = "artifacts\experiments\model_pool_2026_07\keqing_grp_v1\keqing_grp_v1_best.pth"
$ParentCheckpoint = "artifacts\mortal_training\checkpoints\mortal_default_70k_promoted_candidate.pth"
$Manifest = Join-Path $ExpDir "manifest.json"
$AuditPath = Join-Path $ExpDir "reward_ab_audit.json"
$Preflight = Join-Path $ExpDir "grp_reward_preflight_full.json"
$LogDir = Join-Path $ExpDir "pipeline_logs"
$RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Seeds = @(20260721, 20260722, 20260723)
$Groups = @(
    @{ Id = "F_final_rank_mc_weights_only"; Mode = "final_rank_mc" },
    @{ Id = "G_mortal_grp_delta_pt_weights_only"; Mode = "mortal_grp_delta_pt" }
)

if (-not (Test-Path -LiteralPath $Python)) { throw "Windows venv Python is missing: $Python" }
if (-not (Test-Path -LiteralPath $GrpCheckpoint)) { throw "Frozen GRP checkpoint is missing: $GrpCheckpoint" }
if (-not (Test-Path -LiteralPath $ParentCheckpoint)) { throw "Parent checkpoint is missing: $ParentCheckpoint" }
$status = @(git -c core.excludesFile= status --porcelain)
if ($status.Count -ne 0) { throw "Refusing to start epoch3 on a dirty worktree: $($status -join '; ')" }

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Invoke-Logged {
    param([string]$Name, [string[]]$Arguments)
    $LogPath = Join-Path $LogDir "$RunStamp`_$Name.log"
    "[$(Get-Date -Format o)] START $Name" | Tee-Object -FilePath $LogPath -Append
    $PreviousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    & $Python @Arguments 2>&1 | Tee-Object -FilePath $LogPath -Append
    $ExitCode = $LASTEXITCODE
    $ErrorActionPreference = $PreviousErrorAction
    $Error.Clear()
    if ($ExitCode -ne 0) { throw "$Name failed with exit code $ExitCode. See $LogPath" }
    "[$(Get-Date -Format o)] DONE $Name" | Tee-Object -FilePath $LogPath -Append
}

Invoke-Logged "00_prepare_reward_ab" @(
    "training/mortal/prepare_reward_ab.py",
    "--experiment-id", $ExpId,
    "--seeds", ($Seeds -join ","),
    "--grp-checkpoint", $GrpCheckpoint
)

if (-not (Test-Path -LiteralPath $Preflight)) {
    Invoke-Logged "01_reward_preflight" @(
        "training/mortal/preflight_reward_distribution.py",
        "--config", "$ExpDir\G_mortal_grp_delta_pt_weights_only\seed_20260721\config.toml",
        "--output", $Preflight
    )
}

if ($PrepareOnly -or -not $RunTraining) { exit 0 }

$PairSeeds = $Seeds
if ($FirstPairOnly) { $PairSeeds = @($Seeds[0]) }
foreach ($seed in $PairSeeds) {
    foreach ($group in $Groups) {
        $RunDir = Join-Path $ExpDir "$($group.Id)\seed_$seed"
        Invoke-Logged "$($group.Id)_$seed" @(
            "training/run_mortal_dqn_offline.py",
            "--config", "$RunDir\config.toml", "--target-steps", "72000",
            "--device", "cuda", "--num-workers", "0", "--seed", "$seed", "--data-seed", "$seed",
            "--initialize-from", $ParentCheckpoint, "--initial-steps", "70000",
            "--archive-steps", "72000", "--archive-dir", "$RunDir\checkpoints", "--log-every", "50"
        )
    }
}

Invoke-Logged "02_audit_reward_ab" @(
    "training/mortal/audit_reward_ab.py",
    "--root", $ExpDir,
    "--seeds", ($Seeds -join ","),
    "--output", $AuditPath
)
