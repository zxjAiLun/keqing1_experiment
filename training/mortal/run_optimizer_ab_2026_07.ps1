param(
    [switch]$FirstPairOnly,
    [string]$ExperimentId = "optimizer_ab_2026_07_epoch1",
    [int[]]$Seeds = @(20260724, 20260725, 20260726)
)

$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Repo
$Python = Join-Path $Repo ".venv-win\Scripts\python.exe"
$ExpDir = Join-Path $Repo "artifacts\experiments\model_pool_2026_07\$ExperimentId"
$Parent = Join-Path $Repo "artifacts\mortal_training\checkpoints\mortal_default_70k_promoted_candidate.pth"
$LogDir = Join-Path $ExpDir "pipeline_logs"
$ArchiveSteps = "70001,70010,70100,70500,71000,72000"

if (-not (Test-Path -LiteralPath $Python)) { throw "Windows venv Python is missing: $Python" }
if (-not (Test-Path -LiteralPath (Join-Path $ExpDir "manifest.json"))) { throw "Experiment manifest is missing: $ExpDir" }
if (-not (Test-Path -LiteralPath $Parent)) { throw "Parent checkpoint is missing: $Parent" }
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Invoke-Logged {
    param(
        [string]$Name,
        [string[]]$Arguments
    )
    $LogPath = Join-Path $LogDir "$Name.log"
    "[$(Get-Date -Format o)] START $Name" | Tee-Object -FilePath $LogPath -Append
    # Python logging uses stderr on Windows; keep it in the combined log without
    # letting PowerShell's Stop preference terminate on an informational line.
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python @Arguments 2>&1 | Tee-Object -FilePath $LogPath -Append
    $ExitCode = $LASTEXITCODE
    $ErrorActionPreference = $PreviousErrorActionPreference
    if ($ExitCode -ne 0) {
        throw "$Name failed with exit code $ExitCode. See $LogPath"
    }
    "[$(Get-Date -Format o)] DONE $Name" | Tee-Object -FilePath $LogPath -Append
}

$PairSeeds = $Seeds
if ($FirstPairOnly) { $PairSeeds = @($Seeds[0]) }

foreach ($Seed in $PairSeeds) {
    $FreshConfig = Join-Path $ExpDir "A_final_rank_mc_fresh_adam\seed_$Seed\config.toml"
    $PreservedConfig = Join-Path $ExpDir "B_final_rank_mc_preserved_adam\seed_$Seed\config.toml"
    $PreflightOutput = Join-Path $ExpDir "preflight\optimizer_preflight_$Seed.json"
    Invoke-Logged "preflight_$Seed" @(
        "training\mortal\preflight_optimizer_ab.py",
        "--fresh-config", $FreshConfig,
        "--preserved-config", $PreservedConfig,
        "--parent", $Parent,
        "--optimizer-parent", $Parent,
        "--data-seed", "$Seed",
        "--output", $PreflightOutput
    )

    foreach ($Group in @("A_final_rank_mc_fresh_adam", "B_final_rank_mc_preserved_adam")) {
        $RunDir = Join-Path $ExpDir "$Group\seed_$Seed"
        $Config = Join-Path $RunDir "config.toml"
        $State = Join-Path $RunDir "mortal.pth"
        $ArchiveDir = Join-Path $RunDir "checkpoints"
        $Arguments = @(
            "training\run_mortal_dqn_offline.py",
            "--config", $Config,
            "--target-steps", "72000",
            "--device", "cuda",
            "--num-workers", "0",
            "--seed", "$Seed",
            "--data-seed", "$Seed",
            "--archive-steps", $ArchiveSteps,
            "--archive-dir", $ArchiveDir,
            "--log-every", "50"
        )
        if (-not (Test-Path -LiteralPath $State)) {
            $Arguments += @(
                "--initialize-from", $Parent,
                "--initial-steps", "70000"
            )
            if ($Group -eq "B_final_rank_mc_preserved_adam") {
                $Arguments += @("--initialize-optimizer-from", $Parent)
            }
        }
        Invoke-Logged "$Group`_$Seed" $Arguments
    }
}

foreach ($Seed in $PairSeeds) {
    foreach ($Group in @("A_final_rank_mc_fresh_adam", "B_final_rank_mc_preserved_adam")) {
        $State = Join-Path $ExpDir "$Group\seed_$Seed\mortal.pth"
        if (-not (Test-Path -LiteralPath $State)) { throw "missing final checkpoint: $State" }
        foreach ($Step in @(70001, 70010, 70100, 70500, 71000, 72000)) {
            $Archive = Join-Path $ExpDir "$Group\seed_$Seed\checkpoints\mortal_$Step.pth"
            if (-not (Test-Path -LiteralPath $Archive)) { throw "missing archive checkpoint: $Archive" }
        }
    }
}

"optimizer A/B training pipeline completed" | Tee-Object -FilePath (Join-Path $LogDir "pipeline.done")
