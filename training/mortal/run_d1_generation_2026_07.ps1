param(
    [ValidateSet("Smoke", "Shard")]
    [string]$Mode = "Smoke",
    [ValidateRange(0, 23)]
    [int]$ShardIndex = 0,
    [switch]$ResetIncomplete
)

$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Repo
$Python = Join-Path $Repo ".venv-win\Scripts\python.exe"
$Experiment = Join-Path $Repo "artifacts\experiments\model_pool_2026_07\D1_project_owned_population_2026_07"
$DataRoot = Join-Path $Experiment "data"
$Parent = Join-Path $Repo "artifacts\mortal_training\checkpoints\mortal_default_70k_promoted_candidate.pth"
$External = Join-Path $Repo "artifacts\external_mortal_20240308_best_min.pth"
$V3 = Join-Path $Repo "artifacts\experiments\model_pool_2026_07\V3_final_rank_mc_warmstart_2026_07\checkpoints\mortal_74000.pth"
$V2 = Join-Path $Repo "artifacts\experiments\model_pool_2026_07\V2_population_mixed_v4_warmstart_2026_07\checkpoints\mortal_74000.pth"

if (-not (Test-Path -LiteralPath $Python)) { throw "Windows Python is missing: $Python" }
foreach ($Path in @($Parent, $External, $V3, $V2)) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "D1 checkpoint is missing: $Path" }
}
$GitCommit = (& git log -1 --format=%H -- training/mortal/run_d1_generation_2026_07.ps1 training/mortal/audit_d1_dataset.py).Trim()
if ((& git status --porcelain) -join "") { throw "D1 generation requires a clean Git worktree" }
New-Item -ItemType Directory -Force -Path $Experiment, $DataRoot | Out-Null

$ModelPaths = [ordered]@{
    "K0_70k" = $Parent
    "ext_mortal" = $External
    "V3_74000" = $V3
    "V2_74000" = $V2
}
function Get-Sha256([string]$Path) {
    $Hasher = [System.Security.Cryptography.SHA256]::Create()
    try {
        $Stream = [System.IO.File]::OpenRead($Path)
        try {
            return ([System.BitConverter]::ToString($Hasher.ComputeHash($Stream))).Replace("-", "").ToLowerInvariant()
        } finally {
            $Stream.Dispose()
        }
    } finally {
        $Hasher.Dispose()
    }
}
$ModelSha = [ordered]@{}
foreach ($Entry in $ModelPaths.GetEnumerator()) {
    $ModelSha[$Entry.Key] = Get-Sha256 $Entry.Value
}
$MortalRevision = (& git -C (Join-Path $Repo "third_party\Mortal") rev-parse HEAD 2>$null).Trim()
if (-not $MortalRevision) { $MortalRevision = "working-tree" }
$ManifestPath = Join-Path $Experiment "manifest.json"
$ExpectedManifest = [ordered]@{
    schema = "keqing.mortal.d1_generation_manifest.v1"
    experiment_id = "D1_project_owned_population_2026_07"
    git_commit = $GitCommit
    git_dirty = $false
    mortal_revision = $MortalRevision
    models = $ModelPaths
    model_sha256 = $ModelSha
    labels = @("K0_70k", "ext_mortal", "V3_74000", "V2_74000")
    trainable_label = "K0_70k"
    seed_key = 8192
    generation_protocol = "B250"
    rank_points = @(90, 45, 0, -135)
    amp = $false
    seat_mode = "random"
    total_games = 6000
    shard_games = 250
    shard_seed_start = 1600000
    smoke_seed_start = 1599000
}
if (Test-Path -LiteralPath $ManifestPath) {
    $Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
    if ($Manifest.git_commit -ne $GitCommit -or $Manifest.git_dirty -ne $false) {
        $ExistingLogs = @(Get-ChildItem -LiteralPath $DataRoot -Recurse -Filter *.json.gz -ErrorAction SilentlyContinue)
        if ($ExistingLogs.Count -gt 0) {
            throw "existing D1 manifest does not match current clean commit while data already exists"
        }
        $ExpectedManifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $ManifestPath -Encoding UTF8
        $Manifest = $ExpectedManifest
    }
} else {
    $ExpectedManifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $ManifestPath -Encoding UTF8
    $Manifest = $ExpectedManifest
}

if ($Mode -eq "Smoke") {
    $Name = "smoke_25h"
    $Output = Join-Path $DataRoot $Name
    $SeedStart = 1599000
    $Games = 25
    $NativeBatch = 25
} else {
    $Name = "shard_{0:D2}" -f $ShardIndex
    $Output = Join-Path $DataRoot $Name
    $SeedStart = 1600000 + (250 * $ShardIndex)
    $Games = 250
    $NativeBatch = 250
}
$LogDir = Join-Path $Output "logs"
$Existing = @(Get-ChildItem -LiteralPath $LogDir -Filter *.json.gz -ErrorAction SilentlyContinue)
if ($Existing.Count -gt 0 -and $Existing.Count -ne $Games) {
    if (-not $ResetIncomplete) { throw "$Name has $($Existing.Count) logs; pass -ResetIncomplete to discard and rerun the whole shard" }
    $ResolvedData = (Resolve-Path $DataRoot).Path.TrimEnd('\') + '\'
    $ResolvedOutput = (Resolve-Path $Output -ErrorAction SilentlyContinue).Path
    if ($ResolvedOutput -and -not $ResolvedOutput.StartsWith($ResolvedData, [System.StringComparison]::OrdinalIgnoreCase)) { throw "refusing to remove path outside D1 data root: $Output" }
    Remove-Item -LiteralPath $Output -Recurse -Force
}
if ($Existing.Count -eq $Games) {
    & $Python "training\mortal\audit_d1_dataset.py" --data-dir $Output --expected-games $Games --seed-start $SeedStart --seed-key 8192 --output (Join-Path $Output "dataset_audit.json")
    if ($LASTEXITCODE -ne 0) { throw "$Name exists but failed audit; use -ResetIncomplete after inspecting it" }
    Write-Output "$Name already complete and passed audit"
    exit 0
}

$Args = @(
    "training\mortal\four_player_native.py",
    "--model", "K0_70k=$Parent",
    "--model", "ext_mortal=$External",
    "--model", "V3_74000=$V3",
    "--model", "V2_74000=$V2",
    "--output-dir", $Output,
    "--device", "cuda",
    "--require-cuda",
    "--seed-start", "$SeedStart",
    "--seed-key", "8192",
    "--games", "$Games",
    "--seat-mode", "random",
    "--progress-every", "25",
    "--native-batch-games", "$NativeBatch",
    "--rank-points", "90,45,0,-135",
    "--profile"
)
& $Python @Args
if ($LASTEXITCODE -ne 0) { throw "$Name native generation failed with exit code $LASTEXITCODE" }
& $Python "training\mortal\audit_d1_dataset.py" --data-dir $Output --expected-games $Games --seed-start $SeedStart --seed-key 8192 --output (Join-Path $Output "dataset_audit.json")
if ($LASTEXITCODE -ne 0) { throw "$Name failed D1 audit" }
Write-Output "$Name generated and passed audit"
