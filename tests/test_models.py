import numpy as np

from horus_ai.models.isolation_forest import IsolationForestDetector


def test_isolation_forest_fits_scores_predicts_and_round_trips(tmp_path):
    X = np.arange(60, dtype=float).reshape(20, 3)
    detector = IsolationForestDetector(
        n_estimators=20,
        threshold_quantile=0.9,
        random_state=0,
    ).fit(X)

    scores = detector.score(X)
    assert scores.shape == (20,)
    assert np.isfinite(scores).all()
    assert detector.predict(X).shape == (20,)

    path = tmp_path / "detector.joblib"
    detector.save(path)
    loaded = IsolationForestDetector.load(path)
    np.testing.assert_allclose(loaded.score(X), scores)
