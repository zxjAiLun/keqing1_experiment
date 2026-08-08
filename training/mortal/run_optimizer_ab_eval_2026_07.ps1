param(
    [ValidateSet("Screen", "Full")]
    [string]$Stage = "Screen",
    [string]$ExperimentId = "optimizer_ab_2026_07_epoch1",
    [int[]]$Seeds = @(20260724, 20260725, 20260726),
    [int]$SeedStartBase = 940000
)

$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Repo
$Python = Join-Path $Repo ".venv-win\Scripts\python.exe"
$ExpDir = Join-Path $Repo "artifacts\experiments\model_pool_2026_07\$ExperimentId"
$EvalDir = Join-Path $ExpDir "eval_1000h"
$LogDir = Join-Path $EvalDir "eval_logs"
$Parent = Join-Path $Repo "artifacts\mortal_training\checkpoints\mortal_default_70k_promoted_candidate.pth"
$External = Join-Path $Repo "artifacts\external_mortal_20240308_best_min.pth"
$SeedStarts = @{}
for ($Index = 0; $Index -lt $Seeds.Count; $Index++) {
    $SeedStarts[$Seeds[$Index]] = $SeedStartBase + (1000 * $Index)
}
$Games = if ($Stage -eq "Screen") { 250 } else { 1000 }

if (-not (Test-Path -LiteralPath $Python)) { throw "Windows venv Python is missing: $Python" }
if (-not (Test-Path -LiteralPath $Parent)) { throw "70k parent is missing: $Parent" }
if (-not (Test-Path -LiteralPath $External)) { throw "External Mortal reference is missing: $External" }
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Invoke-Logged {
    param(
        [string]$Name,
        [string[]]$Arguments
    )
    $LogPath = Join-Path $LogDir "$Name.log"
    "[$(Get-Date -Format o)] START $Name stage=$Stage games=$Games" | Tee-Object -FilePath $LogPath -Append
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

foreach ($Seed in $Seeds) {
    $RunDir = Join-Path $EvalDir "lineup_$Seed"
    $Fresh = Join-Path $ExpDir "A_final_rank_mc_fresh_adam\seed_$Seed\mortal.pth"
    $Preserved = Join-Path $ExpDir "B_final_rank_mc_preserved_adam\seed_$Seed\mortal.pth"
    if (-not (Test-Path -LiteralPath $Fresh)) { throw "fresh checkpoint is missing: $Fresh" }
    if (-not (Test-Path -LiteralPath $Preserved)) { throw "preserved checkpoint is missing: $Preserved" }
    $Args = @(
        "training\mortal\four_player_native.py",
        "--model", "70k=$Parent",
        "--model", "ext_mortal=$External",
        "--model", "fresh_$Seed=$Fresh",
        "--model", "preserved_$Seed=$Preserved",
        "--output-dir", $RunDir,
        "--device", "cuda",
        "--require-cuda",
        "--seed-start", "$($SeedStarts[$Seed])",
        "--seed-key", "8192",
        "--games", "$Games",
        "--seat-mode", "random",
        "--progress-every", "25",
        "--native-batch-games", "25",
        "--resume"
    )
    Invoke-Logged "lineup_$Seed`_$Stage" $Args
    if ($Stage -eq "Screen") {
        $Snapshot = Join-Path $EvalDir "screen_250h\lineup_$Seed"
        New-Item -ItemType Directory -Force -Path $Snapshot | Out-Null
        foreach ($File in @("metrics.json", "detailed_stats.json", "detailed_stats.md")) {
            $Source = Join-Path $RunDir $File
            if (-not (Test-Path -LiteralPath $Source)) { throw "screen artifact is missing: $Source" }
            Copy-Item -LiteralPath $Source -Destination (Join-Path $Snapshot $File) -Force
        }
        $Platform = Join-Path $RunDir "platform_accounts"
        if (Test-Path -LiteralPath $Platform) {
            Copy-Item -LiteralPath $Platform -Destination (Join-Path $Snapshot "platform_accounts") -Recurse -Force
        }
    }
}

"optimizer A/B eval stage completed: $Stage games=$Games" | Tee-Object -FilePath (Join-Path $LogDir "$Stage.done")
