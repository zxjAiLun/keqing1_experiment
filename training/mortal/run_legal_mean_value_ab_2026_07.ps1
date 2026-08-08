param(
    [switch]$FirstPairOnly,
    [string]$ExperimentId = "legal_mean_value_ab_2026_07",
    [int[]]$Seeds = @(20260803, 20260804, 20260805)
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
    param([string]$Name, [string[]]$Arguments)
    $LogPath = Join-Path $LogDir "$Name.log"
    "[$(Get-Date -Format o)] START $Name" | Tee-Object -FilePath $LogPath -Append
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python @Arguments 2>&1 | Tee-Object -FilePath $LogPath -Append
    $ExitCode = $LASTEXITCODE
    $ErrorActionPreference = $PreviousErrorActionPreference
    if ($ExitCode -ne 0) { throw "$Name failed with exit code $ExitCode. See $LogPath" }
    "[$(Get-Date -Format o)] DONE $Name" | Tee-Object -FilePath $LogPath -Append
}

function Get-Sha256 {
    param([string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-Preflight {
    param(
        [int]$Seed,
        [string]$PreflightPath,
        [string]$ControlConfig,
        [string]$VariantConfig
    )
    if (-not (Test-Path -LiteralPath $PreflightPath)) { throw "missing preflight JSON: $PreflightPath" }
    $Report = Get-Content -LiteralPath $PreflightPath -Raw | ConvertFrom-Json
    if (-not [bool]$Report.passed) { throw "preflight did not pass for seed $Seed" }
    if ([int]$Report.data_seed -ne $Seed) { throw "preflight data seed mismatch for seed $Seed" }
    if (-not [bool]$Report.first_data_batches.identical) { throw "preflight batch hashes differ for seed $Seed" }
    if ([bool]$Report.fingerprints.git_dirty) { throw "preflight recorded dirty git state for seed $Seed" }
    $CurrentCommit = (& git rev-parse HEAD).Trim()
    if ($Report.fingerprints.git_commit -ne $CurrentCommit) { throw "git commit changed after preflight for seed $Seed" }
    $ParentHash = Get-Sha256 $Parent
    if ($Report.fingerprints.parent_sha256 -ne $ParentHash) { throw "parent SHA changed after preflight for seed $Seed" }
    if ($Report.fingerprints.control_config_sha256 -ne (Get-Sha256 $ControlConfig)) { throw "control config changed after preflight for seed $Seed" }
    if ($Report.fingerprints.variant_config_sha256 -ne (Get-Sha256 $VariantConfig)) { throw "variant config changed after preflight for seed $Seed" }
    if ($Report.fingerprints.file_index_sha256 -ne (Get-Sha256 $Report.fingerprints.file_index)) { throw "file index changed after preflight for seed $Seed" }
    foreach ($Entry in @($Report.fingerprints.control_label_files) + @($Report.fingerprints.variant_label_files)) {
        if ($Entry.sha256 -ne (Get-Sha256 $Entry.path)) { throw "label file changed after preflight: $($Entry.path)" }
    }
}

$PairSeeds = $Seeds
if ($FirstPairOnly) { $PairSeeds = @($Seeds[0]) }

foreach ($Seed in $PairSeeds) {
    $ControlConfig = Join-Path $ExpDir "C_behavior_action_mc\seed_$Seed\config.toml"
    $VariantConfig = Join-Path $ExpDir "V_legal_mean_mc\seed_$Seed\config.toml"
    $Preflight = Join-Path $ExpDir "preflight\preflight_$Seed.json"
    Invoke-Logged "preflight_$Seed" @(
        "training\mortal\preflight_legal_mean_objective.py",
        "--control-config", $ControlConfig,
        "--variant-config", $VariantConfig,
        "--parent", $Parent,
        "--data-seed", "$Seed",
        "--output", $Preflight
    )
    Assert-Preflight $Seed $Preflight $ControlConfig $VariantConfig
    foreach ($Group in @("C_behavior_action_mc", "V_legal_mean_mc")) {
        $RunDir = Join-Path $ExpDir "$Group\seed_$Seed"
        $Config = Join-Path $RunDir "config.toml"
        $State = Join-Path $RunDir "mortal.pth"
        $ArchiveDir = Join-Path $RunDir "checkpoints"
        if (-not (Test-Path -LiteralPath $Config)) { throw "missing config: $Config" }
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
                "--initialize-optimizer-from", $Parent,
                "--initial-steps", "70000"
            )
        }
        Invoke-Logged "$Group`_$Seed" $Arguments
    }
}

foreach ($Seed in $PairSeeds) {
    $ControlRun = Join-Path $ExpDir "C_behavior_action_mc\seed_$Seed"
    $VariantRun = Join-Path $ExpDir "V_legal_mean_mc\seed_$Seed"
    $Verification = Join-Path $ExpDir "preflight\verification_$Seed.json"
    $CurrentCommit = (& git rev-parse HEAD).Trim()
    Invoke-Logged "verification_$Seed" @(
        "training\mortal\verify_legal_mean_value_run.py",
        "--run-dir", $ControlRun,
        "--peer-run-dir", $VariantRun,
        "--expected-objective", "behavior_action_mc",
        "--expected-peer-objective", "legal_mean_mc",
        "--expected-seed", "$Seed",
        "--parent", $Parent,
        "--expected-git-commit", $CurrentCommit,
        "--output", $Verification
    )
    $VerificationReport = Get-Content -LiteralPath $Verification -Raw | ConvertFrom-Json
    if (-not [bool]$VerificationReport.passed) { throw "correctness verification failed for seed $Seed" }
    if (-not [bool]$VerificationReport.data_stream_identical) { throw "data stream mismatch for seed $Seed" }
}

foreach ($Seed in $PairSeeds) {
    foreach ($Group in @("C_behavior_action_mc", "V_legal_mean_mc")) {
        $RunDir = Join-Path $ExpDir "$Group\seed_$Seed"
        if (-not (Test-Path -LiteralPath (Join-Path $RunDir "mortal.pth"))) {
            throw "missing final checkpoint: $RunDir"
        }
        foreach ($Step in @(70001, 70010, 70100, 70500, 71000, 72000)) {
            $Archive = Join-Path $RunDir "checkpoints\mortal_$Step.pth"
            if (-not (Test-Path -LiteralPath $Archive)) { throw "missing archive checkpoint: $Archive" }
        }
    }
}

"legal-mean objective A/B training pipeline completed" | Tee-Object -FilePath (Join-Path $LogDir "pipeline.done")
