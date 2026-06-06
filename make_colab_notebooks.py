#!/usr/bin/env python3
"""Convert percent-format Python scripts into lightweight Colab notebooks."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOKS = {
    "investment_finetune_all_in_one_colab.py": "investment_finetune_all_in_one_colab.ipynb",
    "upload_dataset_to_huggingface_colab.py": "upload_dataset_to_huggingface_colab.ipynb",
    "train_unsloth_qwen3_lora_colab.py": "train_unsloth_qwen3_lora_colab.ipynb",
    "run_base_model_test_inference_colab.py": "run_base_model_test_inference_colab.ipynb",
    "run_lora_test_inference_colab.py": "run_lora_test_inference_colab.ipynb",
}


def flush_cell(cells: list[dict], cell_type: str, lines: list[str]) -> None:
    if not lines:
        return
    source = [line if line.endswith("\n") else f"{line}\n" for line in lines]
    if cell_type == "markdown":
        cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
    else:
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source,
            }
        )


def convert(path: Path) -> dict:
    cells: list[dict] = []
    current_type = "code"
    current_lines: list[str] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line.startswith("# %% [markdown]"):
            flush_cell(cells, current_type, current_lines)
            current_type = "markdown"
            current_lines = []
            continue
        if raw_line.startswith("# %%"):
            flush_cell(cells, current_type, current_lines)
            current_type = "code"
            current_lines = []
            continue

        if current_type == "markdown":
            if raw_line.startswith("# "):
                current_lines.append(raw_line[2:])
            elif raw_line == "#":
                current_lines.append("")
            else:
                current_lines.append(raw_line)
        else:
            current_lines.append(raw_line)

    flush_cell(cells, current_type, current_lines)
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    for source_name, notebook_name in NOTEBOOKS.items():
        notebook = convert(ROOT / source_name)
        out = ROOT / notebook_name
        out.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
