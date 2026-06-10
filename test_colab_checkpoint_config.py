#!/usr/bin/env python3
"""Static guard for Colab checkpoint/resume safety settings."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COLAB_SCRIPT = ROOT / "investment_finetune_all_in_one_colab.py"


def main() -> int:
    source = COLAB_SCRIPT.read_text(encoding="utf-8")
    assert "CHECKPOINT_SAVE_STEPS = 10" in source
    assert "ENABLE_HUB_CHECKPOINTS = True" in source
    assert "RESUME_FROM_HUB_CHECKPOINT = True" in source
    assert 'hub_strategy="checkpoint"' in source
    assert "hub_always_push=True" in source
    assert "save_only_model=False" in source
    assert "restore_callback_states_from_checkpoint=True" in source
    assert "snapshot_download(" in source
    assert '"last-checkpoint/**"' in source
    assert "trainer.train(resume_from_checkpoint=resume_checkpoint)" in source

    # Ignore the Colab shell magic cell, then ensure remaining code is valid.
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("!")
    )
    ast.parse(code)
    print("checkpoint config guard passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
