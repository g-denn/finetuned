param(
    [int]$AgentCount = 8,
    [string]$Model = "gpt-5.2"
)

$ErrorActionPreference = "Stop"

$workspace = "C:\Users\Dell\finetuned"
$codex = "C:\Users\Dell\AppData\Roaming\npm\codex.ps1"

for ($i = 1; $i -le $AgentCount; $i++) {
    $batch = Join-Path $workspace ("validation_agent_batch_{0:D2}.json" -f $i)
    if (-not (Test-Path $batch)) {
        Write-Host "Skipping missing batch $batch"
        continue
    }

    $output = Join-Path $workspace ("validation_agent_batch_{0:D2}.result.json" -f $i)
    $stdoutLog = Join-Path $workspace ("validation_agent_batch_{0:D2}.agent.out.log" -f $i)
    $stderrLog = Join-Path $workspace ("validation_agent_batch_{0:D2}.agent.err.log" -f $i)

    $prompt = @"
You are Validation Agent $("{0:D2}" -f $i) for a financial fine-tuning data cleanup project.

Work only on $batch.
Do not modify database rows.
Do not modify any file except your own final output target if needed: $output.
You are not alone in the codebase; do not revert edits made by others.

For each row in $batch, independently research public web evidence and reason through:
- same security identity
- ticker changes or ticker reuse
- acquisition, delisting, bankruptcy, liquidation, or OTC continuation
- splits and reverse splits
- dividends and special dividends
- spin-offs
- whether Yahoo adjusted-close return can be trusted

Treat raw Yahoo math as evidence only, not truth.

Produce a JSON array only in your final answer, one object per row, matching these performance_validation fields:
idea_id, raw_symbol, yahoo_symbol, validation_status, identity_status, corporate_action_status, label_quality,
include_in_training, validated_perf_1y, validated_perf_3y, validated_perf_5y,
split_adjusted, dividend_adjusted, spin_off_adjusted, merger_adjusted,
identity_confidence, return_confidence, validation_reason, corporate_action_timeline,
agent_a_result, agent_b_result, sources, failure_modes.

Agent A must propose evidence. Agent B must be adversarial and set reviewer_status pass/fail.
If evidence is insufficient, mark include_in_training=false and needs_manual_review or insufficient_evidence.
Do not guess.
"@

    $codexArgs = @(
        "exec",
        "-m", $Model,
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C", $workspace,
        "-o", $output,
        $prompt
    )

    Write-Host "Starting Validation Agent $("{0:D2}" -f $i) for $batch"
    $args = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $codex) + $codexArgs
    Start-Process -FilePath "powershell.exe" -ArgumentList $args -WorkingDirectory $workspace -WindowStyle Hidden -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog
}

Write-Host "Started requested validation agents."
