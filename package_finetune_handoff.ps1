$ErrorActionPreference = "Stop"

$OutDir = "C:\Users\Dell\finetuned\dist"
$Zip = Join-Path $OutDir "investment_finetune_handoff.zip"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
if (Test-Path -LiteralPath $Zip) {
    Remove-Item -LiteralPath $Zip -Force
}

$Files = @(
    "FINETUNING_RUNBOOK.md",
    "hf_dataset_README.md",
    "hf_model_README.md",
    "upload_dataset_to_huggingface.py",
    "upload_dataset_to_huggingface_colab.py",
    "upload_dataset_to_huggingface_colab.ipynb",
    "investment_finetune_all_in_one_colab.py",
    "investment_finetune_all_in_one_colab.ipynb",
    "package_hf_dataset_upload_bundle.ps1",
    "train_unsloth_qwen3_lora_colab.py",
    "train_unsloth_qwen3_lora_colab.ipynb",
    "run_base_model_test_inference_colab.py",
    "run_base_model_test_inference_colab.ipynb",
    "run_lora_test_inference_colab.py",
    "run_lora_test_inference_colab.ipynb",
    "fetch_hf_adapter_results.py",
    "check_hf_finetune_status.py",
    "evaluate_outcome_predictions.py",
    "check_finetune_gate.py",
    "train_text_baseline.py",
    "reports\dataset_audit.md",
    "reports\dataset_audit.json",
    "reports\majority_baseline_metrics.json",
    "reports\text_baseline_metrics.json"
)

Compress-Archive -LiteralPath $Files -DestinationPath $Zip -Force
Write-Host "Created $Zip"
