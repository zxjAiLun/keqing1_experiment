param(
    [string]$Python = 'E:\AUbuntuProject\project\keqing1\.venv-win\Scripts\python.exe'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $repoRoot
$experimentRoot = Join-Path $repoRoot 'artifacts/experiments/T1_k0_policy_anchor_continuation_pilot_2026_09'
$runStatus = Join-Path $experimentRoot 'windows_closure_status.json'

function Save-RunStatus([string]$phase, [int]$code) {
    @{
        phase = $phase
        exit_code = $code
        updated_at = (Get-Date).ToUniversalTime().ToString('o')
        runner_pid = $PID
        python = $Python
    } | ConvertTo-Json | Set-Content -LiteralPath $runStatus -Encoding utf8
}

Save-RunStatus 'evaluation_running' 0
& $Python -u training/mortal/eval_t1_k0_policy_anchor_2026_09.py --device cuda --resume
if ($LASTEXITCODE -ne 0) {
    Save-RunStatus 'evaluation_failed' $LASTEXITCODE
    exit $LASTEXITCODE
}
Save-RunStatus 'summary_running' 0
& $Python -u training/mortal/summary_t1_k0_policy_anchor_2026_09.py
if ($LASTEXITCODE -ne 0) {
    Save-RunStatus 'summary_failed' $LASTEXITCODE
    exit $LASTEXITCODE
}
Save-RunStatus 'machine_adjudication_completed_report_pending' 0
