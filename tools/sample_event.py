"""Pick a labelled flow from the TEST split for the Review 2 demo, and optionally send it to the API.

Usage:
    python -m tools.sample_event --label DoS            # print one DoS flow as JSON
    python -m tools.sample_event --label BruteForce --post
    python -m tools.sample_event --list                 # show which labels are available
"""
from __future__ import annotations

import argparse
import json
import urllib.request

from src.ml.common import load_meta, load_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=None, help="class to sample (default: a random attack)")
    ap.add_argument("--post", action="store_true", help="send the flow to the running API")
    ap.add_argument("--url", default="http://127.0.0.1:8000/detect")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    meta = load_meta()
    test = load_table("test")
    if args.list:
        print(test["label"].value_counts().to_string())
        return
    pool = test[test["label"] == args.label] if args.label else test[test["label"] != "Benign"]
    if pool.empty:
        raise SystemExit(f"No test rows for label {args.label!r}. Use --list to see the options.")
    row = pool.sample(1, random_state=args.seed).iloc[0]
    flow = {f: float(row[f]) for f in meta["features"]}
    print(f"True label: {row['label']}  (source file: {row['src_file']}, row {int(row['row_in_file'])})")
    if args.post:
        req = urllib.request.Request(args.url, data=json.dumps(flow).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            print(json.dumps(json.loads(resp.read()), indent=2))
    else:
        print(json.dumps(flow, indent=2))


if __name__ == "__main__":
    main()
