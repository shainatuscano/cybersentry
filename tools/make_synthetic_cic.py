"""Create a SYNTHETIC stand-in for CIC-IDS2017 to smoke-test the pipeline.

The numbers are random. Never report results from this data; it only checks that the scripts run.

Usage:
    CS_WORKDIR=/tmp/cs_smoke python -m tools.make_synthetic_cic
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.ml.common import RAW, SEED, ensure_dirs

FEATS = [" Destination Port", " Flow Duration", " Total Fwd Packets", " Total Backward Packets",
         "Total Length of Fwd Packets", " Total Length of Bwd Packets", " Fwd Packet Length Mean",
         " Bwd Packet Length Mean", "Flow Bytes/s", " Flow Packets/s", " Flow IAT Mean",
         " Fwd IAT Mean", " SYN Flag Count", " ACK Flag Count", " PSH Flag Count",
         " Average Packet Size", " Init_Win_bytes_forward", " Init_Win_bytes_backward",
         "Active Mean", " Idle Mean"]

# raw CIC label -> destination ports it typically uses (None = random high port)
PORTS = {"BENIGN": [80, 443, 53, 22, 8080, 3389, 25], "FTP-Patator": [21], "SSH-Patator": [22],
         "DoS Hulk": [80], "DoS GoldenEye": [80], "DoS slowloris": [80], "DoS Slowhttptest": [80],
         "Heartbleed": [444], "Web Attack \x96 Brute Force": [80], "Web Attack \x96 XSS": [80],
         "Web Attack \x96 Sql Injection": [80], "Infiltration": [444], "Bot": [8080],
         "PortScan": None, "DDoS": [80]}

FILES = {
    "Monday-WorkingHours.pcap_ISCX.csv": {"BENIGN": 30000},
    "Tuesday-WorkingHours.pcap_ISCX.csv": {"BENIGN": 15000, "FTP-Patator": 3000, "SSH-Patator": 2500},
    "Wednesday-workingHours.pcap_ISCX.csv": {"BENIGN": 15000, "DoS Hulk": 5000, "DoS GoldenEye": 1500,
                                             "DoS slowloris": 1500, "DoS Slowhttptest": 1200, "Heartbleed": 11},
    "Thursday-Morning-WebAttacks.pcap_ISCX.csv": {"BENIGN": 12000, "Web Attack \x96 Brute Force": 1500,
                                                  "Web Attack \x96 XSS": 650, "Web Attack \x96 Sql Injection": 21},
    "Thursday-Afternoon-Infilteration.pcap_ISCX.csv": {"BENIGN": 12000, "Infiltration": 36},
    "Friday-Morning.pcap_ISCX.csv": {"BENIGN": 12000, "Bot": 1000},
    "Friday-Afternoon-PortScan.pcap_ISCX.csv": {"BENIGN": 10000, "PortScan": 6000},
    "Friday-Afternoon-DDos.pcap_ISCX.csv": {"BENIGN": 10000, "DDoS": 6000},
}


def class_profile(label: str) -> np.ndarray:
    seed = abs(hash_label(label)) % (2**32)
    return np.random.default_rng(seed).normal(0, 1.1, len(FEATS) - 1)


def hash_label(label: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(label)) * 7919


def make_rows(rng, label: str, n: int, base: np.ndarray) -> pd.DataFrame:
    shift = class_profile(label) if label != "BENIGN" else np.zeros(len(FEATS) - 1)
    run_jitter = rng.normal(0, 0.25, len(FEATS) - 1)  # rows in one run resemble each other
    log_vals = base + shift + run_jitter + rng.normal(0, 0.55, (n, len(FEATS) - 1))
    vals = np.exp(log_vals)
    ports = PORTS[label]
    port = rng.choice(ports, n) if ports else rng.integers(1, 65535, n)
    data = np.column_stack([port, vals])
    df = pd.DataFrame(data, columns=FEATS)
    df[" SYN Flag Count"] = rng.poisson(1.0 + (label == "DDoS") * 3 + (label == "PortScan") * 2, n)
    df[" ACK Flag Count"] = rng.poisson(2.0, n)
    df["Flow Bytes/s"] = df["Flow Bytes/s"] * 50
    return df


def build_file(rng, plan: dict, base: np.ndarray) -> pd.DataFrame:
    runs = []
    for label, total in plan.items():
        left = total
        while left > 0:
            n = min(left, int(rng.integers(300, 1500)))
            runs.append((label, n))
            left -= n
    rng.shuffle(runs)
    parts = [make_rows(rng, lab, n, base).assign(**{" Label": lab}) for lab, n in runs]
    df = pd.concat(parts, ignore_index=True)
    # add the messiness of the real files
    df["Flow ID"] = [f"10.0.0.{i % 250}-{i}" for i in range(len(df))]
    df[" Bwd PSH Flags"] = 0
    df["Fwd Header Length.1"] = df[" Total Fwd Packets"]
    dup = df.sample(frac=0.02, random_state=SEED)
    df = pd.concat([df, dup], ignore_index=True).sort_index(kind="stable")
    n = len(df)
    df.loc[rng.choice(n, int(n * 0.005), replace=False), "Flow Bytes/s"] = np.inf
    df.loc[rng.choice(n, int(n * 0.002), replace=False), " Flow Packets/s"] = np.nan
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RAW))
    ap.add_argument("--scale", type=float, default=1.0, help="multiply all row counts")
    args = ap.parse_args()
    ensure_dirs()
    rng = np.random.default_rng(SEED)
    base = rng.normal(5, 1.0, len(FEATS) - 1)
    from pathlib import Path
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, plan in FILES.items():
        plan = {k: max(1, int(v * args.scale)) if v > 100 else v for k, v in plan.items()}
        df = build_file(rng, plan, base)
        df.to_csv(out / name, index=False, encoding="latin-1")
        print(f"wrote {name}: {len(df):,} rows")


if __name__ == "__main__":
    main()
