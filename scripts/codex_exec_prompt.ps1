param(
    [Parameter(Mandatory = $true)]
    [string]$Workspace,

    [Parameter(Mandatory = $true)]
    [string]$PromptFile
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$promptText = Get-Content -Raw $PromptFile
$CodexCli = "C:\Users\Windows\AppData\Roaming\npm\codex.cmd"

if (-not (Test-Path $CodexCli)) {
    throw "Codex CLI not found: $CodexCli"
}

Push-Location $Workspace
try {
    $promptText | & $CodexCli exec --full-auto --sandbox workspace-write --cd $Workspace -
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
