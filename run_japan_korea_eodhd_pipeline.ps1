param(
    [double]$Sleep = 0.25,
    [int]$Retries = 3,
    [double]$MaxCallsPerMinute = 0,
    [int]$Top = 100,
    [switch]$IncludeDelisted,
    [switch]$SkipUpload
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$OutDir = Join-Path $Root "eodhd_output\japan_korea_fundamentals"
$LogPath = Join-Path $OutDir "pipeline_run.log"
$StatePath = Join-Path $OutDir "pipeline_run_state.json"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Write-RunState {
    param(
        [string]$Stage,
        [string]$Status,
        [int]$ExitCode = 0
    )
    $state = @{
        updated_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        stage = $Stage
        status = $Status
        exit_code = $ExitCode
        sleep = $Sleep
        retries = $Retries
        max_calls_per_minute = $MaxCallsPerMinute
        top = $Top
        include_delisted = [bool]$IncludeDelisted
        skip_upload = [bool]$SkipUpload
    }
    $state | ConvertTo-Json | Set-Content -Encoding UTF8 -Path $StatePath
}

function Invoke-Logged {
    param(
        [string]$Stage,
        [string[]]$Arguments
    )
    Write-RunState -Stage $Stage -Status "running"
    Add-Content -Path $LogPath -Value "$(Get-Date -Format o) START $Stage"
    & $Python @Arguments *>> $LogPath
    $exit = $LASTEXITCODE
    if ($exit -ne 0) {
        Write-RunState -Stage $Stage -Status "failed" -ExitCode $exit
        Add-Content -Path $LogPath -Value "$(Get-Date -Format o) FAILED $Stage exit=$exit"
        exit $exit
    }
    Write-RunState -Stage $Stage -Status "completed"
    Add-Content -Path $LogPath -Value "$(Get-Date -Format o) COMPLETE $Stage"
}

if (-not $env:EODHD_API_TOKEN) {
    throw "Set EODHD_API_TOKEN before running the collection."
}

$collectArgs = @(
    (Join-Path $Root "eodhd_japan_korea_fundamentals.py"),
    "--sleep", "$Sleep",
    "--retries", "$Retries",
    "--max-calls-per-minute", "$MaxCallsPerMinute",
    "--pending-only"
)
if ($IncludeDelisted) {
    $collectArgs += "--include-delisted"
}

Invoke-Logged -Stage "collect_and_normalize" -Arguments $collectArgs

Invoke-Logged -Stage "screen_shareholder_yield" -Arguments @(
    (Join-Path $Root "screen_japan_korea_shareholder_yield.py"),
    "--top", "$Top"
)

Invoke-Logged -Stage "audit_local_dataset" -Arguments @(
    (Join-Path $Root "audit_japan_korea_eodhd_dataset.py"),
    "--require-screening"
)

if (-not $SkipUpload) {
    if (-not $env:HF_TOKEN) {
        throw "Set HF_TOKEN or rerun with -SkipUpload."
    }
    if (-not $env:HF_REPO_ID) {
        throw "Set HF_REPO_ID or rerun with -SkipUpload."
    }
    Invoke-Logged -Stage "upload_huggingface" -Arguments @(
        (Join-Path $Root "upload_japan_korea_fundamentals_to_huggingface.py")
    )
    Invoke-Logged -Stage "audit_uploaded_dataset" -Arguments @(
        (Join-Path $Root "audit_japan_korea_eodhd_dataset.py"),
        "--require-screening",
        "--require-upload-status"
    )
}

Write-RunState -Stage "pipeline" -Status "complete"
Add-Content -Path $LogPath -Value "$(Get-Date -Format o) PIPELINE COMPLETE"
