$ErrorActionPreference = "Stop"

$Python = "C:\Users\Dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

Write-Host "Rebuilding investment fine-tuning dataset..."
& $Python build_investment_finetune_dataset.py

Write-Host "Validating chat JSONL splits..."
& $Python validate_finetune_dataset.py

Write-Host "Generating majority-class baseline..."
& $Python make_baseline_predictions.py

Write-Host "Scoring majority-class baseline..."
& $Python evaluate_outcome_predictions.py reports\majority_baseline_predictions.jsonl --output reports\majority_baseline_metrics.json

Write-Host "Training/scoring TF-IDF text baseline..."
& $Python train_text_baseline.py

Write-Host "Checking Hugging Face auth state..."
$hasEnvToken = [bool]$env:HF_TOKEN -or [bool]$env:HUGGINGFACE_HUB_TOKEN
& $Python -c "from huggingface_hub.utils import get_token; print({'cached_hf_token': bool(get_token())})"
Write-Host (@{ HF_TOKEN_env = $hasEnvToken } | ConvertTo-Json -Compress)

Write-Host "Local fine-tune prep complete."
Write-Host "Next: set HF_TOKEN, run upload_dataset_to_huggingface.py, then train in Colab."
