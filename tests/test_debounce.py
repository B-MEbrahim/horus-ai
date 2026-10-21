"""Tests for horus_ai.evaluation.debounce, using tiny hand-made arrays."""

import numpy as np
import pytest

from horus_ai.evaluation.debounce import AlertDebouncer, debounce
from horus_ai.evaluation.metrics import false_positive_rate, recall


def test_n3_worked_example():
    # idx   : 0 1 2 3 4 5 6 7 8
    # flags : 0 1 0 1 1 1 1 0 1
    # run   : 0 1 0 1 2 3 4 0 1
    # alarm : 0 0 0 0 0 1 1 0 0   (on from the 3rd consecutive flag, off at the first 0)
    flags = [0, 1, 0, 1, 1, 1, 1, 0, 1]
    expected = [0, 0, 0, 0, 0, 1, 1, 0, 0]
    assert debounce(flags, n=3).tolist() == expected


def test_n1_means_no_debouncing():
    flags = [0, 1, 0, 1, 1, 0]
    assert debounce(flags, n=1).tolist() == flags


def test_runs_shorter_than_n_never_alarm():
    flags = [1, 1, 0, 1, 1, 0]  # longest run is 2
    assert debounce(flags, n=3).tolist() == [0, 0, 0, 0, 0, 0]


def test_run_of_exactly_n_alarms_on_last_window_only():
    flags = [0, 1, 1, 1, 0]
    assert debounce(flags, n=3).tolist() == [0, 0, 0, 1, 0]

def test_one_normal_window_resets_the_count():
    # Two runs of 2 separated by a 0: they must NOT add up to 4.
    flags = [1, 1, 0, 1, 1]
    assert debounce(flags, n=3).tolist() == [0, 0, 0, 0, 0]


def test_all_zeros_and_all_ones():
    assert debounce([0, 0, 0, 0], n=2).tolist() == [0, 0, 0, 0]
    assert debounce([1, 1, 1, 1], n=2).tolist() == [0, 1, 1, 1]


def test_empty_input():
    out = debounce([], n=3)
    assert out.shape == (0,)


def test_output_is_int_array_with_same_length():
    out = debounce([0, 1, 1, 1, 0, 1], n=2)
    assert out.dtype == int
    assert out.shape == (6,)


def test_accepts_numpy_bool_array():
    flags = np.array([False, True, True])
    assert debounce(flags, n=2).tolist() == [0, 0, 1]


# ---------------------------------------------------------------- properties of the rule


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_detection_delay_grows_by_n_minus_1_windows(n):
    # First raw flag is at index 2, followed by a long run of 1s.
    # First alarm must be at index 2 + (n - 1).
    flags = [0, 0, 1, 1, 1, 1, 1, 1]
    alarms = debounce(flags, n)
    assert int(np.argmax(alarms)) == 2 + (n - 1)


def test_alarm_is_never_on_where_flag_is_off():
    rng = np.random.default_rng(1)
    flags = rng.integers(0, 2, size=300)
    for n in (1, 2, 3, 5):
        alarms = debounce(flags, n)
        assert np.all(alarms <= flags)


def test_larger_n_gives_a_subset_of_alarms():
    rng = np.random.default_rng(2)
    flags = rng.integers(0, 2, size=300)
    a2 = debounce(flags, 2)
    a3 = debounce(flags, 3)
    a5 = debounce(flags, 5)
    assert np.all(a3 <= a2)
    assert np.all(a5 <= a3)


# ---------------------------------------------------------------- streaming class


def test_streaming_matches_batch_on_the_worked_example():
    flags = [0, 1, 0, 1, 1, 1, 1, 0, 1]
    d = AlertDebouncer(n=3)
    streamed = [int(d.update(f)) for f in flags]
    assert streamed == debounce(flags, n=3).tolist()


@pytest.mark.parametrize("n", [1, 2, 3, 4, 7])
def test_streaming_matches_batch_on_random_data(n):
    rng = np.random.default_rng(n)
    flags = rng.integers(0, 2, size=500)
    d = AlertDebouncer(n)
    streamed = [int(d.update(int(f))) for f in flags]
    assert streamed == debounce(flags, n).tolist()


def test_run_length_property_counts_consecutive_flags():
    d = AlertDebouncer(n=3)
    assert d.run_length == 0
    d.update(1)
    d.update(1)
    assert d.run_length == 2
    d.update(0)
    assert d.run_length == 0


def test_reset_clears_the_run():
    d = AlertDebouncer(n=3)
    d.update(1)
    d.update(1)
    d.reset()
    assert d.run_length == 0
    # After a reset we need a full N flags again.
    assert d.update(1) is False
    assert d.update(1) is False
    assert d.update(1) is True


def test_update_returns_plain_bool_and_accepts_bool_input():
    d = AlertDebouncer(n=1)
    assert d.update(True) is True
    assert d.update(False) is False


def test_two_debouncers_do_not_share_state():
    # One debouncer per tower: tower A's flags must not affect tower B.
    tower_a = AlertDebouncer(n=2)
    tower_b = AlertDebouncer(n=2)
    tower_a.update(1)
    assert tower_b.update(1) is False  # B has only seen one flag itself
    assert tower_a.update(1) is True


# ---------------------------------------------------------------- input validation


@pytest.mark.parametrize("bad_n", [0, -1, 2.5, "3", None, True])
def test_bad_n_raises(bad_n):
    with pytest.raises(ValueError, match="n must be"):
        AlertDebouncer(bad_n)
    with pytest.raises(ValueError, match="n must be"):
        debounce([0, 1, 1], bad_n)


def test_numpy_integer_n_is_accepted():
    assert debounce([1, 1, 1], np.int64(2)).tolist() == [0, 1, 1]


def test_non_binary_flags_raise():
    with pytest.raises(ValueError, match="0/1"):
        debounce([0, 1, 2], n=2)


def test_non_1d_flags_raise():
    with pytest.raises(ValueError, match="1-D"):
        debounce([[0, 1], [1, 1]], n=2)


def test_streaming_rejects_bad_flag_and_keeps_state():
    d = AlertDebouncer(n=3)
    d.update(1)
    with pytest.raises(ValueError, match="0/1"):
        d.update(2)
    assert d.run_length == 1  # the bad flag did not corrupt the counter


# ---------------------------------------------------------------- debounce + metrics together


def test_debouncing_trades_recall_for_fewer_false_alarms():
    # idx    : 0 1 2 3 4 5 6 7
    # y_true : 0 0 0 0 0 1 1 1
    # raw    : 0 1 0 0 0 1 1 1   <- one isolated false alarm at idx 1
    y_true = [0, 0, 0, 0, 0, 1, 1, 1]
    raw = [0, 1, 0, 0, 0, 1, 1, 1]
    alarms = debounce(raw, n=2)  # -> 0 0 0 0 0 0 1 1

    assert alarms.tolist() == [0, 0, 0, 0, 0, 0, 1, 1]

    # Raw: FP=1, TN=4 -> FPR 0.2.  Debounced: no false alarm -> FPR 0.
    assert false_positive_rate(y_true, raw) == pytest.approx(0.2)
    assert false_positive_rate(y_true, alarms) == 0.0

    # The price: the first attack window (idx 5) is now missed -> recall 2/3.
    assert recall(y_true, raw) == pytest.approx(1.0)
    assert recall(y_true, alarms) == pytest.approx(2 / 3)