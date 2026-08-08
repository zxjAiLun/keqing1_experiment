param(
    [ValidateRange(0, 23)]
    [int]$StartShard = 1,
    [ValidateRange(0, 23)]
    [int]$EndShard = 23,
    [switch]$ResetIncomplete
)

$ErrorActionPreference = "Stop"
if ($StartShard -gt $EndShard) { throw "StartShard must not exceed EndShard" }
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Repo
$Launcher = Join-Path $Repo "training\mortal\run_d1_generation_2026_07.ps1"

for ($Shard = $StartShard; $Shard -le $EndShard; $Shard++) {
    Write-Output ("[d1] starting shard_{0:D2}" -f $Shard)
    $ChildArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $Launcher,
        "-Mode", "Shard",
        "-ShardIndex", "$Shard"
    )
    if ($ResetIncomplete) { $ChildArgs += "-ResetIncomplete" }
    & powershell.exe @ChildArgs
    if ($LASTEXITCODE -ne 0) {
        throw ("D1 shard_{0:D2} failed with exit code {1}; later shards were not started" -f $Shard, $LASTEXITCODE)
    }
    Write-Output ("[d1] shard_{0:D2} passed" -f $Shard)
}

Write-Output ("[d1] shards {0:D2}-{1:D2} completed" -f $StartShard, $EndShard)
