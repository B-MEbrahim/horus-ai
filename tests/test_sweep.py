"""Tests for horus_ai.evaluation.sweep, using the same 12-window run as test_report.py.

Shared run (window_seconds = 1, window i ends at t = i + 1):

    idx        : 0    1    2    3    4     5     6     7     8    9    10   11
    y_true     : 0    0    0    1    1     1     1     1     0    0    0    0
    offset ns  : 0    0    0    300  700   1100  1600  2000  0    0    0    0
    score      : 0.1  0.9  0.2  0.4  0.5   0.8   0.9   0.7   0.3  0.1  0.6  0.2

The attack is idx 3..7 (starts at t = 3.0 s) and the failure is at t = 7.0 s (idx 6).

Hand-worked rows used below:

    threshold 0.5, N=2 : alarms idx 5-7        -> F1 0.75,  delay 3.0 s, lead +1.0 s (in time)
    threshold 0.8, N=1 : flags  idx 1, 5, 6    -> P 2/3, R 0.4, F1 0.5, FPR 1/7,
                                                  delay 3.0 s, lead +1.0 s, 1 false episode
    threshold 0.8, N=2 : alarm  idx 6 only     -> P 1.0, R 0.2, F1 1/3,
                                                  delay 4.0 s, lead 0.0 s (too late)
"""

import numpy as np
import pytest

from horus_ai.evaluation.report import evaluate_run
from horus_ai.evaluation.sweep import select_operating_point, sweep, threshold_grid

Y_TRUE = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
OFFSETS = [0, 0, 0, 300, 700, 1100, 1600, 2000, 0, 0, 0, 0]
SCORES = [0.1, 0.9, 0.2, 0.4, 0.5, 0.8, 0.9, 0.7, 0.3, 0.1, 0.6, 0.2]
TIMES = np.arange(1, 13)


def run_sweep(**overrides):
    """sweep() on the shared run; override any argument by keyword."""
    kwargs = dict(
        y_true=Y_TRUE,
        times=TIMES,
        offsets_ns=OFFSETS,
        window_seconds=1.0,
        scores=SCORES,
    )
    kwargs.update(overrides)
    return sweep(**kwargs)


def one_row(threshold, n, **overrides):
    return run_sweep(thresholds=[threshold], n_values=(n,), **overrides)[0]


# ---------------------------------------------------------------- threshold_grid


def test_grid_is_evenly_spaced_quantiles():
    # 5 scores, 5 candidates: quantiles 0, 0.25, 0.5, 0.75, 1 land exactly on the scores.
    assert threshold_grid([0, 1, 2, 3, 4], num=5).tolist() == [0, 1, 2, 3, 4]
    assert threshold_grid([0, 1, 2, 3, 4], num=3).tolist() == [0, 2, 4]


def test_grid_spans_from_the_minimum_to_the_maximum_score():
    grid = threshold_grid(SCORES, num=10)
    assert grid[0] == pytest.approx(0.1)  # flags every window
    assert grid[-1] == pytest.approx(0.9)  # flags only the top windows


def test_grid_is_sorted_and_has_no_duplicates():
    # Many repeated scores -> the quantiles repeat -> duplicates must be removed.
    grid = threshold_grid([1, 1, 1, 1, 1, 1, 5], num=10)
    assert grid.tolist() == sorted(set(grid.tolist()))
    assert len(grid) <= 10


def test_grid_with_constant_scores_has_one_candidate():
    assert threshold_grid([2.0, 2.0, 2.0], num=5).tolist() == [2.0]


def test_grid_works_for_any_score_scale():
    # Same shape of data, very different scale (e.g. a reconstruction error in ns).
    small = threshold_grid([0.1, 0.2, 0.3], num=3)
    large = threshold_grid([1000.0, 2000.0, 3000.0], num=3)
    assert small.tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert large.tolist() == pytest.approx([1000.0, 2000.0, 3000.0])


def test_grid_rejects_bad_input():
    with pytest.raises(ValueError, match="non-empty 1-D"):
        threshold_grid([])
    with pytest.raises(ValueError, match="non-empty 1-D"):
        threshold_grid([[0.1, 0.2], [0.3, 0.4]])
    with pytest.raises(ValueError, match="finite"):
        threshold_grid([0.1, np.nan, 0.3])
    with pytest.raises(ValueError, match="finite"):
        threshold_grid([0.1, np.inf, 0.3])
    with pytest.raises(ValueError, match="num"):
        threshold_grid([0.1, 0.2], num=1)


# ---------------------------------------------------------------- sweep: shape and order


def test_one_row_per_threshold_and_n_pair():
    rows = run_sweep(thresholds=[0.5, 0.8], n_values=(1, 2, 3))
    assert len(rows) == 6


def test_row_order_is_threshold_first_then_n():
    rows = run_sweep(thresholds=[0.5, 0.8], n_values=(1, 2))
    assert [(r["threshold"], r["n_consecutive"]) for r in rows] == [
        (0.5, 1),
        (0.5, 2),
        (0.8, 1),
        (0.8, 2),
    ]


def test_threshold_and_n_are_the_first_two_columns_and_not_repeated():
    row = one_row(0.5, 2)
    assert list(row)[:2] == ["threshold", "n_consecutive"]
    assert "config.threshold" not in row
    assert "config.n_consecutive" not in row


def test_row_has_the_flat_report_columns():
    row = one_row(0.5, 2)
    for key in (
        "config.n_windows",
        "raw.f1",
        "alarm.f1",
        "false_alarms.episodes_per_hour",
        "detection.mean_delay_s",
        "lead_time.protection_rate",
    ):
        assert key in row


def test_default_n_values_are_1_2_3_5():
    rows = run_sweep(thresholds=[0.5])
    assert [r["n_consecutive"] for r in rows] == [1, 2, 3, 5]


def test_a_single_number_is_accepted_as_thresholds():
    rows = run_sweep(thresholds=0.5, n_values=(1,))
    assert len(rows) == 1
    assert rows[0]["threshold"] == 0.5


def test_default_thresholds_come_from_threshold_grid():
    grid = threshold_grid(SCORES)
    rows = run_sweep(n_values=(1, 2))
    assert len(rows) == len(grid) * 2
    assert sorted({r["threshold"] for r in rows}) == pytest.approx(grid.tolist())


# ---------------------------------------------------------------- sweep: the numbers


def test_row_matches_a_direct_evaluate_run_call():
    direct = evaluate_run(
        Y_TRUE,
        TIMES,
        OFFSETS,
        1.0,
        scores=SCORES,
        threshold=0.8,
        n_consecutive=2,
    ).to_flat_dict()
    expected = {
        k: v for k, v in direct.items() if k not in {"config.threshold", "config.n_consecutive"}
    }

    row = one_row(0.8, 2)
    assert row.pop("threshold") == 0.8
    assert row.pop("n_consecutive") == 2
    assert row == expected


def test_hand_worked_row_threshold_05_n2():
    row = one_row(0.5, 2)
    assert row["alarm.f1"] == pytest.approx(0.75)
    assert row["false_alarms.episodes"] == 0
    assert row["detection.mean_delay_s"] == pytest.approx(3.0)
    assert row["lead_time.mean_lead_s"] == pytest.approx(1.0)
    assert row["lead_time.protection_rate"] == pytest.approx(1.0)


def test_hand_worked_row_threshold_08_n1():
    row = one_row(0.8, 1)  # flags idx 1, 5, 6 (a score of exactly 0.8 counts)
    assert row["alarm.precision"] == pytest.approx(2 / 3)
    assert row["alarm.recall"] == pytest.approx(0.4)
    assert row["alarm.f1"] == pytest.approx(0.5)
    assert row["alarm.fpr"] == pytest.approx(1 / 7)
    assert row["false_alarms.episodes"] == 1
    assert row["detection.mean_delay_s"] == pytest.approx(3.0)
    assert row["lead_time.mean_lead_s"] == pytest.approx(1.0)
    assert row["lead_time.protection_rate"] == pytest.approx(1.0)


def test_hand_worked_row_threshold_08_n2_is_too_late():
    row = one_row(0.8, 2)  # run idx 5-6 reaches 2 at idx 6 -> alarm at t = 7.0 = failure
    assert row["alarm.precision"] == pytest.approx(1.0)
    assert row["alarm.recall"] == pytest.approx(0.2)
    assert row["alarm.f1"] == pytest.approx(1 / 3)
    assert row["false_alarms.episodes"] == 0
    assert row["detection.mean_delay_s"] == pytest.approx(4.0)
    assert row["lead_time.mean_lead_s"] == pytest.approx(0.0)
    assert row["lead_time.protection_rate"] == pytest.approx(0.0)


def test_raising_the_threshold_never_adds_alarms():
    # Higher threshold -> the flags are a subset -> the alarms are a subset too,
    # so recall and FPR can only stay the same or go down.
    thresholds = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0]
    rows = run_sweep(thresholds=thresholds, n_values=(2,))
    recalls = [r["alarm.recall"] for r in rows]
    fprs = [r["alarm.fpr"] for r in rows]
    assert all(a >= b for a, b in zip(recalls, recalls[1:]))
    assert all(a >= b for a, b in zip(fprs, fprs[1:]))


def test_stride_is_passed_through_to_evaluate_run():
    default = one_row(0.5, 1)
    overlapping = one_row(0.5, 1, stride_seconds=0.5)
    assert default["false_alarms.episodes_per_hour"] == pytest.approx(7200 / 7)
    assert overlapping["false_alarms.episodes_per_hour"] == pytest.approx(14400 / 7)
    assert overlapping["alarm.f1"] == default["alarm.f1"]


def test_limit_is_passed_through_to_evaluate_run():
    # Limit 1000 ns -> failure at idx 5 (t = 6.0); the N=2 alarm is also at t = 6.0 -> too late.
    row = one_row(0.5, 2, limit_ns=1000)
    assert row["config.limit_ns"] == 1000.0
    assert row["lead_time.n_too_late"] == 1
    assert row["lead_time.protection_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------- sweep: validation


def test_empty_thresholds_raise():
    with pytest.raises(ValueError, match="thresholds must not be empty"):
        run_sweep(thresholds=[])


def test_empty_n_values_raise():
    with pytest.raises(ValueError, match="n_values must not be empty"):
        run_sweep(thresholds=[0.5], n_values=())


def test_bad_n_raises():
    with pytest.raises(ValueError, match="n must be"):
        run_sweep(thresholds=[0.5], n_values=(0,))


def test_bad_scores_raise_when_thresholds_are_chosen_automatically():
    bad = list(SCORES)
    bad[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        run_sweep(scores=bad)


def test_errors_from_evaluate_run_still_surface():
    with pytest.raises(ValueError, match="length mismatch"):
        run_sweep(thresholds=[0.5], scores=SCORES[:-1])
    with pytest.raises(ValueError, match="stride_seconds"):
        run_sweep(thresholds=[0.5], stride_seconds=0)


# ---------------------------------------------------------------- select_operating_point


def row(eph, rate, f1, **extra):
    """A minimal hand-made row with only the three columns the selection rule reads."""
    return {
        "false_alarms.episodes_per_hour": eph,
        "lead_time.protection_rate": rate,
        "alarm.f1": f1,
        **extra,
    }


def test_budget_excludes_rows_with_too_many_false_alarms():
    a = row(eph=0.5, rate=0.5, f1=0.5, id="A")
    b = row(eph=2.0, rate=1.0, f1=0.9, id="B")  # better, but over budget
    assert select_operating_point([a, b], max_false_episodes_per_hour=1.0)["id"] == "A"


def test_budget_boundary_is_inclusive():
    a = row(eph=0.5, rate=0.5, f1=0.5, id="A")
    b = row(eph=2.0, rate=1.0, f1=0.9, id="B")
    assert select_operating_point([a, b], max_false_episodes_per_hour=2.0)["id"] == "B"


def test_zero_budget_allows_only_rows_with_no_false_alarms():
    clean = row(eph=0.0, rate=0.3, f1=0.3, id="clean")
    noisy = row(eph=0.1, rate=1.0, f1=1.0, id="noisy")
    assert select_operating_point([noisy, clean], max_false_episodes_per_hour=0.0)["id"] == "clean"


def test_returns_none_when_nothing_fits_the_budget():
    rows = [row(eph=3.0, rate=1.0, f1=1.0), row(eph=5.0, rate=1.0, f1=1.0)]
    assert select_operating_point(rows, max_false_episodes_per_hour=1.0) is None


def test_returns_none_for_empty_rows():
    assert select_operating_point([], max_false_episodes_per_hour=1.0) is None


def test_protection_rate_beats_f1():
    protects = row(eph=0.0, rate=0.9, f1=0.1, id="protects")
    accurate = row(eph=0.0, rate=0.5, f1=0.99, id="accurate")
    assert select_operating_point([accurate, protects], 1.0)["id"] == "protects"


def test_tie_on_protection_rate_goes_to_higher_f1():
    c = row(eph=1.0, rate=0.8, f1=0.6, id="C")
    d = row(eph=1.0, rate=0.8, f1=0.7, id="D")
    assert select_operating_point([c, d], 5.0)["id"] == "D"


def test_tie_on_rate_and_f1_goes_to_fewer_false_alarms():
    e = row(eph=0.2, rate=0.8, f1=0.7, id="E")
    f = row(eph=0.1, rate=0.8, f1=0.7, id="F")
    assert select_operating_point([e, f], 5.0)["id"] == "F"


def test_missing_protection_rate_counts_as_the_lowest():
    # None = no attack reached the limit, so nothing was protected or measured.
    unknown = row(eph=0.0, rate=None, f1=0.99, id="unknown")
    zero = row(eph=0.0, rate=0.0, f1=0.1, id="zero")
    assert select_operating_point([unknown, zero], 1.0)["id"] == "zero"


def test_a_lone_row_with_missing_protection_rate_can_still_be_chosen():
    only = row(eph=0.0, rate=None, f1=0.5, id="only")
    assert select_operating_point([only], 1.0)["id"] == "only"


def test_returns_the_original_row_and_does_not_change_the_input():
    rows = [row(eph=0.0, rate=0.5, f1=0.5, id="A"), row(eph=0.0, rate=0.9, f1=0.5, id="B")]
    snapshot = [dict(r) for r in rows]
    chosen = select_operating_point(rows, 1.0)
    assert chosen is rows[1]
    assert rows == snapshot


def test_negative_budget_raises():
    with pytest.raises(ValueError, match="max_false_episodes_per_hour"):
        select_operating_point([row(0.0, 1.0, 1.0)], -0.1)


# ---------------------------------------------------------------- sweep + selection together


def test_selection_on_the_real_sweep_with_a_zero_budget():
    # Rows with 0 false-alarm episodes: (0.5, 2) rate 1.0, (0.5, 3) rate 0.0 (alarm at t = 7.0),
    # (0.8, 2) rate 0.0, (0.8, 3) rate 0.0 (never alarms). The N=1 rows have false alarms.
    rows = run_sweep(thresholds=[0.5, 0.8], n_values=(1, 2, 3))
    best = select_operating_point(rows, max_false_episodes_per_hour=0.0)
    assert (best["threshold"], best["n_consecutive"]) == (0.5, 2)
    assert best["lead_time.protection_rate"] == pytest.approx(1.0)
    assert best["false_alarms.episodes"] == 0


def test_selection_on_the_real_sweep_with_a_generous_budget():
    # Now (0.5, 1), (0.5, 2) and (0.8, 1) all protect the attack (rate 1.0);
    # the tie goes to the highest F1: 0.75 for (0.5, 2).
    rows = run_sweep(thresholds=[0.5, 0.8], n_values=(1, 2, 3))
    best = select_operating_point(rows, max_false_episodes_per_hour=1e9)
    assert (best["threshold"], best["n_consecutive"]) == (0.5, 2)
    assert best["alarm.f1"] == pytest.approx(0.75)