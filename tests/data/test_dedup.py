import numpy as np
import pandas as pd

from narutocv.data import dedup
from narutocv.data.manifest import parse_uuid1, session_id_for


def test_hamming_matrix_known():
    a = np.array([0b0000, 0b1111], dtype=np.uint64)
    b = np.array([0b0001, 0b1111, 0xFFFFFFFFFFFFFFFF], dtype=np.uint64)
    d = dedup.hamming_matrix(a, b)
    assert d.tolist() == [[1, 4, 64], [3, 0, 60]]


def test_cluster_by_hamming_threshold():
    h = np.array([0, 0b11, 0b111111, 0xFFFFFFFFFFFFFFFF], dtype=np.uint64)
    # dist(0,1)=2, dist(1,2)=4, dist(0,2)=6
    assert len(set(dedup.cluster_by_hamming(h, 2))) == 3  # {0,1},{2},{3}
    lab = dedup.cluster_by_hamming(h, 4)  # chain 0-1-2 joins
    assert lab[0] == lab[1] == lab[2] != lab[3]


def test_parse_uuid1():
    node, ts = parse_uuid1("IMG_e8a16238-547b-11ea-879a-48f17fc25591.png")
    assert node == "48f17fc25591"
    assert (ts.year, ts.month, ts.day) == (2020, 2, 21)
    assert parse_uuid1("IMG_0001.png") is None
    assert session_id_for("nouuid", None) == "nouuid"
    assert session_id_for(node, ts) == "48f17fc25591_2020-02-21"


def test_leakage_vs_train_ignores_aug():
    h = "0000000000000000"
    prep = pd.DataFrame(
        [
            {"prep_split": "train", "cls": "a", "is_aug": False, "phash": "ffffffffffffffff"},
            {"prep_split": "train", "cls": "a", "is_aug": True, "phash": h},  # aug must be ignored
            {"prep_split": "test", "cls": "a", "is_aug": False, "phash": h},
            {"prep_split": "val", "cls": "a", "is_aug": False, "phash": "fffffffffffffff0"},
        ]
    )
    r = dedup.leakage_vs_train(prep, thresholds=(0, 6))
    assert r["test"]["same_class_le_6_pct"] == 0.0  # only the aug matched
    assert r["val"]["same_class_le_6_pct"] == 100.0  # distance 4 to the original
    assert r["val"]["same_class_le_0_pct"] == 0.0
