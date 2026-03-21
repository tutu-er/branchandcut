param(
    [Parameter(Mandatory = $true)]
    [string]$Workspace,

    [Parameter(Mandatory = $true)]
    [string]$PromptFile
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$promptText = Get-Content -Raw $PromptFile

Push-Location $Workspace
try {
    codex exec --full-auto --sandbox workspace-write --cd $Workspace $promptText
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
