# Data backup manifest

This repository keeps source code, runbooks, audit reports, compact status
manifests, and the core processed investment datasets in GitHub. Large generated
dataset mirrors are kept in private Hugging Face dataset repositories so they
can be restored without turning this public GitHub repository into a 20+ GB
data store.

## GitHub

- Repository: `g-denn/finetuned`
- SQL database dump: `VIC_IDEAS.sql`
- Storage: Git LFS
- LFS object: `sha256:9a7f32f44db9d73b25226f0ba051d141ad04e112496e0dee42e98a38464084e0`
- Size: 203,115,709 bytes

The processed investment files under `data/processed/`, model evaluation
predictions, and large ZIP/CSV/JSONL artifacts already committed to this
repository also use Git LFS according to `.gitattributes`.

## Private Hugging Face datasets

The local upload status files in `eodhd_output/` are the machine-readable source
of truth for the checks below.

| Dataset | Local status evidence | Recorded verification |
| --- | --- | --- |
| `Gden/eodhd-vic-financial-data` | `hf_extracted_upload_status.json`, `hf_combined_table_upload_status.json` | Complete; 9,119 remote files; combined table 4,541 rows |
| `Gden/eodhd-japan-korea-fundamentals` | `hf_japan_korea_fundamentals_upload_status.json` | Complete; no expected files missing |
| `Gden/vic-pitch-financial-context-eodhd` | `hf_vic_pitch_financial_context_upload_status.json`, `hf_raw_vic_audit_upload_status.json` | Complete; 8,341 rows in six shards; audit files recorded |
| `Gden/vic-pitch-financial-context-clean-sft` | `hf_clean_vic_finetune_upload_status.json` | Complete; 5,249 rows; remote summary matches local |
| `Gden/vic-pitch-financial-context-repaired-clean-sft` | `hf_repaired_clean_vic_upload_status.json` | 5,639 rows; remote summary matches local |
| `Gden/vic-pitch-financial-context-repaired-clean-transcripts` | `hf_transcript_enriched_vic_upload_status.json` | 5,639 rows; remote summary matches local |

These datasets are private. Do not copy their contents into this public GitHub
repository without a deliberate licensing/privacy review.

## Deliberately local and reproducible

The following paths are ignored by Git because they are dependency caches,
redundant upload staging trees, or local mirrors of the private datasets above:

- `.codex_deps/`
- `dist/eodhd_dataset_financial_pull.zip`
- `eodhd_output/alpha_vantage_transcripts/`
- `eodhd_output/dataset_financial_pull/`
- `eodhd_output/japan_korea_fundamentals/`
- `eodhd_output/public_transcript_sources/`
- `eodhd_output/raw_vic_public_transcript_companion/`
- `eodhd_output/vic_pitch_financial_context*/`

Before deleting any local data, verify the corresponding live remote repository
and retain this manifest plus the matching `hf_*_upload_status.json` file.
