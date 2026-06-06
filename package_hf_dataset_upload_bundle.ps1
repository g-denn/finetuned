$ErrorActionPreference = "Stop"

$Root = "C:\Users\Dell\finetuned"
$OutDir = Join-Path $Root "dist"
$Zip = Join-Path $OutDir "hf_dataset_upload_bundle.zip"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
if (Test-Path -LiteralPath $Zip) {
    Remove-Item -LiteralPath $Zip -Force
}

$RelativeFiles = @(
    "data\processed\investment_train.jsonl",
    "data\processed\investment_val.jsonl",
    "data\processed\investment_test.jsonl",
    "data\processed\investment_canonical.jsonl",
    "hf_dataset_README.md",
    "FINETUNING_RUNBOOK.md",
    "reports\dataset_audit.md",
    "reports\dataset_audit.json",
    "reports\majority_baseline_metrics.json",
    "reports\majority_baseline_predictions.jsonl",
    "reports\text_baseline_metrics.json",
    "reports\text_baseline_test_predictions.jsonl",
    "reports\text_baseline_val_predictions.jsonl"
)

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$Archive = [System.IO.Compression.ZipFile]::Open($Zip, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($RelativePath in $RelativeFiles) {
        $Source = Join-Path $Root $RelativePath
        if (-not (Test-Path -LiteralPath $Source)) {
            throw "Missing expected file: $Source"
        }
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $Archive,
            $Source,
            ($RelativePath -replace "\\", "/"),
            [System.IO.Compression.CompressionLevel]::Optimal
        ) | Out-Null
    }
}
finally {
    $Archive.Dispose()
}
Write-Host "Created $Zip"
