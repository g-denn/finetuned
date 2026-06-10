# %% [markdown]
# # Upload VIC Investment Dataset To Hugging Face
#
# Run this in Google Colab if local Hugging Face login/OAuth is broken.
#
# First create `dist/hf_dataset_upload_bundle.zip` locally, then upload that zip
# into the Colab file browser. This notebook extracts the bundle, creates a
# private Hugging Face dataset repo, and uploads the processed train/val/test
# files plus audit reports. Baselines are computed inside the training notebook.

# %%
!pip install -q --upgrade pip
!pip install -q "huggingface_hub"

# %%
import os
import zipfile
from getpass import getpass
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_file, whoami

if not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = getpass("Paste a Hugging Face write token: ")

HF_USERNAME = "YOUR_HF_USERNAME"
DATASET_REPO_NAME = "vic-investment-outcomes-sft"
BUNDLE_ZIP = Path("/content/hf_dataset_upload_bundle.zip")
EXTRACT_DIR = Path("/content/hf_dataset_upload_bundle")

# %%
if not BUNDLE_ZIP.exists():
    raise FileNotFoundError(
        "Upload dist/hf_dataset_upload_bundle.zip into Colab first, "
        "then rerun this cell."
    )

if EXTRACT_DIR.exists():
    import shutil

    shutil.rmtree(EXTRACT_DIR)
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(BUNDLE_ZIP) as archive:
    archive.extractall(EXTRACT_DIR)

sorted(str(path.relative_to(EXTRACT_DIR)) for path in EXTRACT_DIR.rglob("*") if path.is_file())

# %%
if HF_USERNAME == "YOUR_HF_USERNAME":
    HF_USERNAME = whoami(token=os.environ["HF_TOKEN"])["name"]

DATASET_REPO = f"{HF_USERNAME}/{DATASET_REPO_NAME}"
create_repo(
    DATASET_REPO,
    repo_type="dataset",
    private=True,
    token=os.environ["HF_TOKEN"],
    exist_ok=True,
)
DATASET_REPO

# %%
uploads = [
    ("data/processed/investment_train.jsonl", "investment_train.jsonl"),
    ("data/processed/investment_val.jsonl", "investment_val.jsonl"),
    ("data/processed/investment_test.jsonl", "investment_test.jsonl"),
    ("data/processed/investment_canonical.jsonl", "investment_canonical.jsonl"),
    ("hf_dataset_README.md", "README.md"),
    ("FINETUNING_RUNBOOK.md", "FINETUNING_RUNBOOK.md"),
    ("reports/dataset_audit.md", "reports/dataset_audit.md"),
    ("reports/dataset_audit.json", "reports/dataset_audit.json"),
]

for local_name, repo_name in uploads:
    local_path = EXTRACT_DIR / local_name
    if not local_path.exists():
        raise FileNotFoundError(local_path)
    upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=repo_name,
        repo_id=DATASET_REPO,
        repo_type="dataset",
        token=os.environ["HF_TOKEN"],
    )
    print(f"uploaded {repo_name}")

# %%
api = HfApi(token=os.environ["HF_TOKEN"])
info = api.dataset_info(DATASET_REPO)
print(f"Private dataset uploaded: https://huggingface.co/datasets/{info.id}")
