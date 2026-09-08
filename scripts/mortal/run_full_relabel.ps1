# Full-corpus external relabel launcher (18,000 hanchans: S0 6000 + D3 6000 + V2 6000).
#
# Official frozen parameters (validated by the dedicated 60-hanchan resource smoke,
# commit a47f6ab):
#   --rows-per-shard 2048      single-writer buffer 269 MiB, no concatenate double-buffer
#   --inference-batch 512      1024 hits a deterministic VRAM cliff on the 8GB WDDM GPU
#   fp32 (no --enable-amp)
#
# Pre-start checks are intentionally thin: input dirs exist, total .json.gz == 18000,
# no other CUDA compute task.  Resume is supported: if the manifest already exists,
# the relabel tool continues from the last snapshot (pass -Resume).

param(
    [string]$OutputDir = "artifacts/experiments/student_policy_v1/labels_full_18000h",
    [string]$Python = "E:\AUbuntuProject\project\keqing1\.venv-win\Scripts\python.exe",
    [switch]$Resume
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..\..   # repo root (scripts/mortal is two levels below)

$s0Logs = "E:\AUbuntuProject\project\keqing1\artifacts\experiments\model_pool_2026_07\S0_pure_ext_selfplay_6000h\logs"
$v2Root = "E:\AUbuntuProject\project\keqing1\artifacts\experiments\model_pool_2026_07\V2_data"
$d3Root = "E:\AUbuntuProject\project\keqing1_experiment\artifacts\experiments\model_pool_2026_07\D3_uncertainty_guided_exploration_2026_08"

# --- Check 1: input directories exist -------------------------------------
$v2Pools = @(
    "$v2Root\v4_70k_t1_v0b_2000h\logs",
    "$v2Root\v4_70k_v1_80k_2000h\logs",
    "$v2Root\v4_v0b_v1_t1_2000h\logs"
)
$missing = @()
if (-not (Test-Path $s0Logs)) { $missing += $s0Logs }
foreach ($p in $v2Pools) { if (-not (Test-Path $p)) { $missing += $p } }
# D3 shards live under two parents: shard_000 in generation_production (the frozen
# B250 gate shard) and shard_001..023 in generation_continuation.
$d3Shards = @()
foreach ($parent in @("generation_production", "generation_continuation")) {
    $d3Shards += Get-ChildItem -Path (Join-Path $d3Root $parent) -Directory -Filter "shard_*"
}
if ($d3Shards.Count -ne 24) {
    Write-Error "Expected 24 D3 shard directories, found $($d3Shards.Count)"
    exit 1
}
foreach ($s in $d3Shards) { if (-not (Test-Path (Join-Path $s.FullName "logs"))) { $missing += "$($s.FullName)\logs" } }
if ($missing.Count -gt 0) {
    Write-Error "Missing input directories:`n$($missing -join "`n")"
    exit 1
}

# --- Check 2: total .json.gz count == 18000 --------------------------------
$s0Count = (Get-ChildItem $s0Logs -Filter *.json.gz).Count
$d3Count = 0
foreach ($s in $d3Shards) { $d3Count += (Get-ChildItem (Join-Path $s.FullName 'logs') -Filter *.json.gz).Count }
$v2Count = 0
foreach ($p in $v2Pools) { $v2Count += (Get-ChildItem $p -Filter *.json.gz).Count }
$total = $s0Count + $d3Count + $v2Count
Write-Host "S0=$s0Count D3=$d3Count V2=$v2Count total=$total"
if ($total -ne 18000) {
    Write-Error "Expected 18000 total .json.gz logs, found $total"
    exit 1
}

# --- Check 3: no other CUDA compute task -----------------------------------
# On this WDDM desktop the compute-apps list also contains system graphics
# processes (ShellHost, NVIDIA Overlay, explorer, StartMenu).  Only flag real
# compute workloads: python or other training/inference executables.
$gpuProcs = @(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>$null |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_ -match ',\s*(python|pythonw|uv|python\.exe|torchrun)' })
if ($gpuProcs.Count -gt 0) {
    Write-Error "Other CUDA compute tasks are running:`n$($gpuProcs -join "`n")`nrelabel must run on an exclusive GPU."
    exit 1
}

# --- Build and print the exact command --------------------------------------
$poolArgs = @("--pool", "S0=$s0Logs")
foreach ($s in $d3Shards) { $poolArgs += @("--pool", "D3=$($s.FullName)\logs") }
foreach ($p in $v2Pools) { $poolArgs += @("--pool", "V2=$p") }

$argList = @(
    "training/mortal/relabel_ext_teacher.py",
    "--output-dir", $OutputDir
) + $poolArgs + @(
    "--rows-per-shard", "2048",
    "--inference-batch", "512",
    "--manifest-snapshot-every", "100"
)
if ($Resume) { $argList += "--resume" }

Write-Host ""
Write-Host "python $($argList -join ' ')"
Write-Host ""

& $Python @argList
exit $LASTEXITCODE
