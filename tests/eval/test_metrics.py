import numpy as np
import pytest

from narutocv.eval import metrics as M


def onehot(labels, n):
    p = np.zeros((len(labels), n))
    p[np.arange(len(labels)), labels] = 1.0
    return p


def test_perfect():
    y = [0, 1, 2, 0, 1, 2]
    m = M.compute_metrics(y, onehot(y, 3), ["a", "b", "c"])
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["ece"] == 0.0
    assert np.trace(np.array(m["confusion_matrix"])) == 6


def test_all_wrong():
    y = [0, 1, 2, 0]
    pred = [1, 2, 0, 1]
    m = M.compute_metrics(y, onehot(pred, 3), ["a", "b", "c"])
    assert m["accuracy"] == 0.0
    assert m["macro_f1"] == 0.0
    # confidently wrong: conf=1, acc=0 -> ECE = 1
    assert m["ece"] == pytest.approx(1.0)


def test_hand_computed_prf():
    # true: a a a b b c ; pred: a a b b c c
    y = [0, 0, 0, 1, 1, 2]
    pred = [0, 0, 1, 1, 2, 2]
    cm = M.confusion_matrix(y, pred, 3)
    assert cm.tolist() == [[2, 1, 0], [0, 1, 1], [0, 0, 1]]
    p, r, f1, s = M.per_class_prf(cm)
    assert p.tolist() == pytest.approx([1.0, 0.5, 0.5])
    assert r.tolist() == pytest.approx([2 / 3, 0.5, 1.0])
    assert f1.tolist() == pytest.approx([0.8, 0.5, 2 / 3])
    assert s.tolist() == [3, 2, 1]
    assert M.macro_f1_from_cm(cm) == pytest.approx((0.8 + 0.5 + 2 / 3) / 3)
    assert M.accuracy(y, pred) == pytest.approx(4 / 6)


def test_macro_f1_ignores_absent_classes():
    y = [0, 0, 1, 1]
    cm = M.confusion_matrix(y, [0, 0, 1, 1], 3)  # class 2 has no support
    assert M.macro_f1_from_cm(cm) == 1.0


def test_known_ece():
    # 4 samples, 2 classes.
    # conf .9 correct, conf .9 wrong, conf .6 correct, conf .6 correct
    probs = np.array([[0.9, 0.1], [0.9, 0.1], [0.4, 0.6], [0.4, 0.6]])
    y = [0, 1, 1, 1]
    # bin (0.8,0.867]? 15 bins: width 1/15. 0.9 -> bin 13, 0.6 -> bin 8 (0.533,0.6]
    # bin 13: n=2 acc=.5 conf=.9 -> |0.5-0.9|*2/4 = 0.2
    # bin 8 : n=2 acc=1 conf=.6 -> |1-0.6|*2/4 = 0.2
    assert M.ece(probs, y) == pytest.approx(0.4)


def test_ece_bin_edge_convention():
    # conf exactly 0.5 with 2 bins -> (0, .5] -> bin 0; conf 1.0 -> last bin
    probs = np.array([[0.5, 0.5], [1.0, 0.0]])
    _, tab = M.ece_bins(probs, [0, 0], n_bins=2)
    assert tab["counts"].tolist() == [1, 1]


def test_bootstrap_perfect_has_degenerate_ci():
    y = np.array([0] * 5 + [1] * 5)
    ci = M.bootstrap_ci(y, onehot(y, 2), 2, n_boot=50, seed=1)
    assert ci["accuracy"] == [1.0, 1.0]
    assert ci["macro_f1"] == [1.0, 1.0]


def test_bootstrap_seeded_and_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 60)
    probs = rng.dirichlet(np.ones(3), 60)
    a = M.bootstrap_ci(y, probs, 3, n_boot=200, seed=7)
    b = M.bootstrap_ci(y, probs, 3, n_boot=200, seed=7)
    assert a == b
    acc = M.accuracy(y, probs.argmax(1))
    assert a["accuracy"][0] <= acc <= a["accuracy"][1]


def test_worst_classes_and_plots(tmp_path):
    y = [0, 0, 1, 1, 2, 2]
    pred = [0, 0, 1, 0, 1, 1]
    m = M.compute_metrics(y, onehot(pred, 3), ["a", "b", "c"])
    assert M.worst_classes(m, 1)[0][0] == "c"
    M.reliability_diagram(onehot(pred, 3), y, tmp_path / "r.png")
    M.confusion_matrix_png(m["confusion_matrix"], ["a", "b", "c"], tmp_path / "c.png")
    assert (tmp_path / "r.png").stat().st_size > 0
    assert (tmp_path / "c.png").stat().st_size > 0
