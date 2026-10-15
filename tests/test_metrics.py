"""Tests for horus_ai.evaluation.metrics, using tiny hand-made arrays.
"""

import numpy as np
import pytest


from horus_ai.evaluation.metrics import (
    ConfusionCounts,
    confusion_counts,
    f1_score,
    false_alarms_per_hour,
    false_positive_rate,
    precision,
    recall,
)


def test_confusion_counts_one_of_each():
    # idx: 0  1  2  3
    # true 0  0  1  1
    # pred 0  1  1  0
    #      TN FP TP FN
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 0]
    assert confusion_counts(y_true, y_pred) == ConfusionCounts(tp=1, fp=1, tn=1, fn=1)


def test_confusion_counts_sum_to_number_of_windows():
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 1, 0, 0]
    c = confusion_counts(y_true, y_pred)
    assert c.tp + c.fp + c.tn + c.fn == len(y_true)


def test_all_metrics_on_one_of_each_example():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 0]
    # TP=1 FP=1 TN=1 FN=1
    assert precision(y_true, y_pred) == 0.5 
    assert recall(y_true, y_pred) == 0.5
    assert f1_score(y_true, y_pred) == 0.5
    assert false_positive_rate(y_true, y_pred) == pytest.approx(0.5)


def test_all_metrics_on_uneven_example():
    # true 0 0 0 0 1 1 1 1
    # pred 0 0 0 1 1 1 0 0
    # TP=2 (idx 4,5)  FP=1 (idx 3)  TN=3 (idx 0,1,2)  FN=2 (idx 6,7)
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 1, 0, 0]
    assert precision(y_true, y_pred) == pytest.approx(2 / 3)
    assert recall(y_true, y_pred) == pytest.approx(2 / 4)
    assert f1_score(y_true, y_pred) == pytest.approx(4 / 7)
    assert false_positive_rate(y_true, y_pred) == pytest.approx(1 / 4)


def test_perfect_detector():
    y = [0, 0, 1, 1, 0, 1]
    assert precision(y, y) == 1.0
    assert recall(y, y) == 1.0
    assert f1_score(y, y) == 1.0
    assert false_positive_rate(y, y) == 0.0


def test_completely_wrong_detector():
    y_true = [0, 1]
    y_pred = [1, 0]  # TP=0 FP=1 TN=0 FN=1
    assert precision(y_true, y_pred) == 0.0
    assert recall(y_true, y_pred) == 0.0
    assert f1_score(y_true, y_pred) == 0.0
    assert false_positive_rate(y_true, y_pred) == 1.0


def test_detector_that_never_fires():
    # TP=0 FP=0 -> precision is 0/0, defined as 0.0
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 0, 0]
    assert precision(y_true, y_pred) == 0.0
    assert recall(y_true, y_pred) == 0.0
    assert f1_score(y_true, y_pred) == 0.0
    assert false_positive_rate(y_true, y_pred) == 0.0


def test_run_with_no_attacks():
    # No real attack windows -> recall is 0/0, defined as 0.0. FPR still works.
    y_true = [0, 0, 0, 0]
    y_pred = [0, 1, 0, 0]
    assert recall(y_true, y_pred) == 0.0
    assert precision(y_true, y_pred) == 0.0  # TP=0, FP=1
    assert false_positive_rate(y_true, y_pred) == pytest.approx(0.25)


def test_run_with_only_attacks_has_zero_fpr():
    # No normal windows -> FPR is 0/0, defined as 0.0
    y_true = [1, 1, 1]
    y_pred = [1, 0, 1]
    assert false_positive_rate(y_true, y_pred) == 0.0


def test_accepts_numpy_arrays_and_bools():
    y_true = np.array([False, False, True, True])
    y_pred = np.array([0, 1, 1, 0])
    assert confusion_counts(y_true, y_pred) == ConfusionCounts(tp=1, fp=1, tn=1, fn=1)


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        precision([0, 1, 1], [0, 1])

        
def test_non_binary_values_raise():
    with pytest.raises(ValueError, match="0/1"):
        recall([0, 1, 2], [0, 1, 1])
    with pytest.raises(ValueError, match="0/1"):
        recall([0, 1, 1], [0, 1, -1])


def test_non_1d_input_raises():
    with pytest.raises(ValueError, match="1-D"):
        f1_score([[0, 1], [1, 0]], [[0, 1], [1, 0]])


def test_false_alarms_per_hour_basic():
    # 4 normal windows x 900 s = 1 hour of normal operation, 1 false alarm
    y_true = [0, 0, 0, 0]
    y_pred = [1, 0, 0, 0]
    assert false_alarms_per_hour(y_true, y_pred, 900) == pytest.approx(1.0)


def test_false_alarms_per_hour_longer_windows():
    # 4 normal windows x 1800 s = 2 hours, 1 false alarm -> 0.5 per hour
    y_true = [0, 0, 0, 0]
    y_pred = [1, 0, 0, 0]
    assert false_alarms_per_hour(y_true, y_pred, 1800) == pytest.approx(0.5)


def test_false_alarms_per_hour_ignores_attack_windows():
    # Only the 2 normal windows count as normal time: 2 x 1800 s = 1 hour.
    # The flag on the attack window (idx 2) is a correct alarm, not a false one.
    y_true = [0, 0, 1, 1]
    y_pred = [1, 0, 1, 1]
    assert false_alarms_per_hour(y_true, y_pred, 1800) == pytest.approx(1.0)


def test_false_alarms_per_hour_no_normal_windows():
    assert false_alarms_per_hour([1, 1], [1, 1], 60) == 0.0


@pytest.mark.parametrize("bad_seconds", [0, -5])
def test_false_alarms_per_hour_rejects_bad_window_length(bad_seconds):
    with pytest.raises(ValueError, match="seconds_per_window"):
        false_alarms_per_hour([0, 1], [0, 1], bad_seconds)  


def test_matches_sklearn_on_random_data():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_pred = rng.integers(0, 2, size=200)

    assert precision(y_true, y_pred) == pytest.approx(sk.precision_score(y_true, y_pred))
    assert recall(y_true, y_pred) == pytest.approx(sk.recall_score(y_true, y_pred))
    assert f1_score(y_true, y_pred) == pytest.approx(sk.f1_score(y_true, y_pred))

    tn, fp, fn, tp = sk.confusion_matrix(y_true, y_pred).ravel()
    assert confusion_counts(y_true, y_pred) == ConfusionCounts(
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn)
    )                                                  