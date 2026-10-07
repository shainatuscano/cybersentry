"""Choose which trained classifier the detection service uses, overriding the automatic selection.

Usage:
    python -m src.ml.stage3.set_detector xgb --reason "XGBoost leads on validation without Infiltration"
    python -m src.ml.stage3.set_detector --auto      # back to the rule's choice

The override is recorded in detector_meta.json (selection_reason starts with "manual override"),
so the API health endpoint and the docs can show which model is serving and why.
"""
from __future__ import annotations

import argparse
import json

from .config import MODEL_DIR, RESULT_DIR, read_detector_meta, update_detector_meta
from .models import MODEL_NAMES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", choices=list(MODEL_NAMES))
    ap.add_argument("--reason", default="")
    ap.add_argument("--auto", action="store_true", help="restore the model chosen by the selection rule")
    args = ap.parse_args()
    meta = read_detector_meta()
    if args.auto:
        sel = json.loads((RESULT_DIR / "stage3_summary.json").read_text())["selection"]
        update_detector_meta(selected_model=sel["selected_model"], selection_reason=sel["reason"])
        print(f"Serving {sel['selected_model']} (selection rule)")
        return
    if not args.model:
        ap.error("give a model name or --auto")
    if not (MODEL_DIR / f"{args.model}.joblib").exists():
        raise SystemExit(f"{MODEL_DIR / (args.model + '.joblib')} not found: train it first")
    update_detector_meta(selected_model=args.model,
                         selection_reason=f"manual override (was {meta.get('selected_model')}): {args.reason}".strip())
    print(f"Serving {args.model}; restart the API to load it")


if __name__ == "__main__":
    main()
