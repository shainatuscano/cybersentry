import numpy as np
import pandas as pd

from src.ml.common import decide, decide_many, group_label
from src.ml.prepare_data import blocked_split


def test_group_label_maps_cic_names():
    assert group_label("BENIGN") == "Benign"
    assert group_label("DoS Hulk") == "DoS"
    assert group_label("DoS slowloris") == "DoS"
    assert group_label("Heartbleed") == "DoS"
    assert group_label("DDoS") == "DDoS"
    assert group_label("FTP-Patator") == "BruteForce"
    assert group_label("SSH-Patator") == "BruteForce"
    assert group_label("Web Attack \x96 Brute Force") == "WebAttack"
    assert group_label("Web Attack � XSS") == "WebAttack"
    assert group_label("Web Attack \x96 Sql Injection") == "WebAttack"
    assert group_label("PortScan") == "PortScan"
    assert group_label("Bot") == "Botnet"
    assert group_label("Infiltration") == "Infiltration"


def test_decide_rules():
    assert decide("DoS", 0.95, 0.1, 0.5) == (True, "known_attack")
    assert decide("DoS", 0.40, 0.9, 0.5)[0] is True
    assert decide("DoS", 0.40, 0.1, 0.5)[0] is False
    assert decide("Benign", 0.99, 0.9, 0.5) == (True, "unknown_anomaly")
    assert decide("Benign", 0.99, 0.1, 0.5) == (False, "benign")


def test_decide_many_matches_scalar():
    rng = np.random.default_rng(0)
    pred = rng.integers(0, 3, 200)
    prob = rng.random(200)
    anom = rng.random(200)
    classes = ["Benign", "A", "B"]
    vec = decide_many(pred, prob, anom, 0.7)
    scalar = [decide(classes[p], pr, an, 0.7)[0] for p, pr, an in zip(pred, prob, anom)]
    assert vec.tolist() == scalar


def test_blocked_split_keeps_blocks_whole_and_all_classes():
    n = 20000
    df = pd.DataFrame({
        "src_file": ["f1"] * n,
        "row_in_file": np.arange(n),
        "label": ["Benign"] * 12000 + ["DoS"] * 8000,
    })
    split = blocked_split(df, block_size=500)
    block = df["row_in_file"] // 500
    # every block lands in exactly one split (classes are large enough to use block assignment)
    assert (pd.DataFrame({"b": block, "s": split}).groupby("b")["s"].nunique() == 1).all()
    for lab in ("Benign", "DoS"):
        assert set(split[df["label"] == lab]) == {"train", "val", "test"}


def test_blocked_split_small_class_is_chronological():
    df = pd.DataFrame({
        "src_file": ["f1"] * 6000,
        "row_in_file": np.arange(6000),
        "label": ["Benign"] * 5900 + ["Rare"] * 100,
    })
    split = blocked_split(df, block_size=1000)
    rare = split[df["label"] == "Rare"].tolist()
    assert rare == ["train"] * 70 + ["val"] * 15 + ["test"] * 15
