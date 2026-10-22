"""tests/test_public_api.py: the package exposes what it promises."""

import horus_ai.evaluation as ev


def test_every_name_in_all_exists():
    for name in ev.__all__:
        assert hasattr(ev, name), f"{name} is in __all__ but not defined"


def test_all_has_no_duplicates():
    assert len(ev.__all__) == len(set(ev.__all__))


def test_private_helpers_are_not_exported():
    assert not any(name.startswith("_") for name in ev.__all__)


def test_evaluate_run_works_through_the_short_import_path():
    report = ev.evaluate_run(
        y_true=[0, 1, 1, 0],
        times=[1, 2, 3, 4],
        offsets_ns=[0, 800, 1600, 0],
        window_seconds=1.0,
        flags=[0, 1, 1, 0],
    )
    assert isinstance(report, ev.EvaluationReport)
    assert report.alarm["f1"] == 1.0