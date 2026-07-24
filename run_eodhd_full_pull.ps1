param(
    [int]$StartOffset = 150,
    [int]$EndOffset = 4541,
    [int]$ChunkSize = 100,
    [int]$TotalSymbols = 4541,
    [double]$Sleep = 0.05,
    [int]$Retries = 2,
    [string]$WorkerName = "main"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$Puller = Join-Path $Root "eodhd_dataset_financial_pull.py"
$OutDir = Join-Path $Root "eodhd_output\dataset_financial_pull"
$LogPath = Join-Path $OutDir "full_pull_$WorkerName.log"
$StatePath = Join-Path $OutDir "full_pull_runner_state_$WorkerName.json"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Write-Log {
    param([string]$Message)
    $line = "$(Get-Date -Format o) $Message"
    Add-Content -Path $LogPath -Value $line
    Write-Output $line
}

if (-not $env:EODHD_API_TOKEN) {
    throw "Set EODHD_API_TOKEN before running the full pull."
}

if ($EndOffset -gt $TotalSymbols) {
    $EndOffset = $TotalSymbols
}

Write-Log "Starting EODHD pull worker=$WorkerName StartOffset=$StartOffset EndOffset=$EndOffset ChunkSize=$ChunkSize TotalSymbols=$TotalSymbols"

for ($offset = $StartOffset; $offset -lt $EndOffset; $offset += $ChunkSize) {
    $remaining = $EndOffset - $offset
    $limit = [Math]::Min($ChunkSize, $remaining)
    $chunkState = @{
        updated_at = (Get-Date).ToUniversalTime().ToString("o")
        status = "running"
        worker = $WorkerName
        offset = $offset
        limit = $limit
        end_offset = $EndOffset
        total_symbols = $TotalSymbols
    }
    $chunkState | ConvertTo-Json | Set-Content -Encoding UTF8 -Path $StatePath
    Write-Log "Chunk start offset=$offset limit=$limit"

    & $Python $Puller --limit $limit --offset $offset --sleep $Sleep --retries $Retries --checklist-every 10 *>> $LogPath
    $exit = $LASTEXITCODE
    if ($exit -ne 0) {
        $chunkState.status = "failed"
        $chunkState.exit_code = $exit
        $chunkState.updated_at = (Get-Date).ToUniversalTime().ToString("o")
        $chunkState | ConvertTo-Json | Set-Content -Encoding UTF8 -Path $StatePath
        Write-Log "Chunk failed offset=$offset exit=$exit"
        exit $exit
    }

    $chunkState.status = "completed"
    $chunkState.updated_at = (Get-Date).ToUniversalTime().ToString("o")
    $chunkState | ConvertTo-Json | Set-Content -Encoding UTF8 -Path $StatePath
    Write-Log "Chunk complete offset=$offset limit=$limit"
}

& $Python $Puller --status-only *>> $LogPath
Write-Log "EODHD pull worker=$WorkerName complete."
