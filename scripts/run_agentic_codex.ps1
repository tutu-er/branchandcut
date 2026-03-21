param(
    [string]$SampleRange = "210:215",
    [int]$NCheck = 1,
    [int]$NTest = 5,
    [int]$FpIter = 3,
    [string[]]$TargetFiles = @("src/feasibility_pump.py"),
    [int]$MaxIters = 6,
    [int]$StagnationLimit = 3,
    [int]$EvalTimeoutSec = 1800,
    [int]$AgentTimeoutSec = 900,
    [string]$ResultsDir = "result/agentic_fp_optimizer",
    [string]$Goal = "",
    [switch]$NoBcdSurrogate,
    [switch]$PrintOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$CodexWrapper = Join-Path $PSScriptRoot "codex_exec_prompt.ps1"
$CodexCli = "C:\Users\Windows\AppData\Roaming\npm\codex.cmd"

Push-Location $RepoRoot
try {
    $null = Get-Command python -ErrorAction Stop
    if (-not (Test-Path $CodexWrapper)) {
        throw "Missing wrapper script: $CodexWrapper"
    }
    if (-not (Test-Path $CodexCli)) {
        throw "Missing Codex CLI: $CodexCli"
    }

    $evalParts = @(
        "python"
        "tests/eval_feasibility_pump.py"
        "--sample-range", $SampleRange
        "--n-check", $NCheck.ToString()
        "--n-test", $NTest.ToString()
        "--fp-iter", $FpIter.ToString()
    )
    if ($NoBcdSurrogate) {
        $evalParts += "--no-bcd-surrogate"
    }
    $evalCommand = [string]::Join(" ", $evalParts)

    $agentTemplate = "powershell -NoProfile -ExecutionPolicy Bypass -File `"$CodexWrapper`" -Workspace `"{workspace}`" -PromptFile `"{prompt_file}`""

    $pythonArgs = @(
        "agentic_fp_optimizer.py"
        "--results-dir", $ResultsDir
        "--eval-command", $evalCommand
        "--agent-kind", "command"
        "--agent-command-template", $agentTemplate
        "--eval-timeout-sec", $EvalTimeoutSec.ToString()
        "--agent-timeout-sec", $AgentTimeoutSec.ToString()
        "--max-iters", $MaxIters.ToString()
        "--stagnation-limit", $StagnationLimit.ToString()
    )

    if ($Goal.Trim()) {
        $pythonArgs += @("--goal", $Goal)
    }

    foreach ($target in $TargetFiles) {
        $pythonArgs += @("--target-files", $target)
    }

    Write-Host "Workspace: $RepoRoot"
    Write-Host "Eval command: $evalCommand"
    Write-Host "Target files: $($TargetFiles -join ', ')"
    Write-Host "Max iters: $MaxIters"
    Write-Host "Stagnation limit: $StagnationLimit"
    Write-Host "Eval timeout sec: $EvalTimeoutSec"
    Write-Host "Agent timeout sec: $AgentTimeoutSec"
    if ($NoBcdSurrogate) {
        Write-Host "Mode: subproblem surrogate only"
    } else {
        Write-Host "Mode: BCD + subproblem unified surrogate"
    }

    if ($PrintOnly) {
        Write-Host ""
        Write-Host "python arguments:"
        foreach ($arg in $pythonArgs) {
            Write-Host "  $arg"
        }
        exit 0
    }

    & python @pythonArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
