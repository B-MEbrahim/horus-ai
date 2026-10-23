# horus_ai.evaluation

Judges a spoofing detector the way the 5G network needs it judged: not just "is it accurate?" but **"did it raise the alarm before the clocks drifted past the 1.5 µs limit, without crying wolf?"**

Pure NumPy. No model or network dependencies, so it works with any detector (Isolation Forest, LSTM autoencoder, EKF, ...).

## The pipeline

```
scores --(>= threshold)--> raw flags --debounce(N)--> alarms --> metrics
```

- A **score** is the detector's output for one window (higher = more anomalous).
- A **raw flag** says "this window looks anomalous".
- An **alarm** is raised only after **N consecutive** anomalous windows. This is what the SDN controller would act on, so alarms are what get graded.

## What you must provide

One entry per window, in time order:

| Input | Meaning |
|---|---|
| `y_true` | 0/1 ground truth (1 = the window is under attack) |
| `times` | **end** time of each window, in seconds (strictly increasing) |
| `offsets_ns` | the **true** clock offset in ns at the end of each window (ground truth from the data generator, not seen by the detector) |
| `scores` | detector output, **higher = more anomalous** |

> **Score direction.** Some libraries use the opposite sign. sklearn's `IsolationForest.score_samples()` is *lower* for anomalies, so pass `-score_samples(X)`. An autoencoder's reconstruction error already has the right direction.

## Quick start

```python
from horus_ai.evaluation import evaluate_run

report = evaluate_run(
    y_true=y_true,
    times=times,
    offsets_ns=offsets_ns,
    window_seconds=10.0,
    scores=scores,
    threshold=0.5,
    n_consecutive=3,          # alarm after 3 anomalous windows in a row
)

report.alarm["f1"]
report.detection["mean_delay_s"]
report.lead_time["protection_rate"]
report.to_flat_dict()         # one row for a comparison table
```

You can pass `flags=` (0/1) instead of `scores=` and `threshold=`, but never both.

## What the report contains

| Section | Contents |
|---|---|
| `config` | settings used (N, threshold, limit, window counts) |
| `raw` | precision, recall, F1, FPR on the flags **before** debouncing |
| `alarm` | the same four metrics on the debounced alarms |
| `false_alarms` | episodes, episodes per hour, windows per hour |
| `detection` | attacks caught, event recall, mean/median/max detection delay |
| `lead_time` | in-time / too-late / missed counts, protection rate, lead statistics |

## Choosing the threshold and N

```python
from horus_ai.evaluation import sweep, select_operating_point

rows = sweep(y_true, times, offsets_ns, 10.0, scores, n_values=(1, 2, 3, 5))
best = select_operating_point(rows, max_false_episodes_per_hour=1.0)
```

`sweep` returns one flat dict per (threshold, N) pair (`pandas.DataFrame(rows)` makes it a table). `select_operating_point` stays within a false-alarm budget, then maximizes the protection rate (ties: higher F1, then fewer false alarms).

**Run the sweep on validation data, then evaluate the chosen (threshold, N) once on held-out test data.** Picking the best row and reporting it on the same data gives optimistic numbers.

## Conventions (fixed, so results are comparable)

- Every window is judged at its **end** time, the moment its verdict is known.
- An **attack event** is a run of consecutive windows with `y_true == 1`. Its start time is the start of its first window.
- **Detection delay** = end time of the first alarm window inside the event − event start. The best possible delay is one window length.
- A **missed** attack has delay `None`, never 0. Alarms before an attack are false alarms, not negative delays.
- **Failure time** = end time of the first window inside the attack where `|offset| >= limit_ns` (default 1500 ns).
- **Lead time** = failure time − alarm time. Positive is good. **Zero counts as too late** (the failover would have no time to act).
- A false-alarm **episode** is a burst of consecutive false-alarm windows. A burst counts once, because it causes one needless failover.
- Division by zero returns `0.0` (for example precision of a detector that never fires). Protection rate is `None` when no attack reached the limit.

## Known limits

- **Sudden jumps can never be caught in time.** If the offset jumps past 1.5 µs inside the first attack window, even a perfect detector has lead time ≤ 0. Lead time matters most for gradual drift, so report sudden-jump and drift attacks separately.
- **Debouncing costs time.** Each extra window of N adds one window length of delay and removes one window length of lead time.
- If an alarm from a false-alarm burst is still on when an attack begins, the delay is reported as one window length.
- If windows overlap, pass `stride_seconds` (time between window starts) so the "per hour" rates are right. It defaults to `window_seconds`.

## Files

| File | Purpose |
|---|---|
| `metrics.py` | confusion counts, precision, recall, F1, FPR, false alarms per hour |
| `debounce.py` | N-consecutive rule: `debounce()` (batch) and `AlertDebouncer` (streaming, one per tower in the live service) |
| `events.py` | attack events, detection delay, false-alarm episodes |
| `lead_time.py` | was the alarm before the failure limit? |
| `report.py` | `evaluate_run()` chains everything into one `EvaluationReport` |
| `sweep.py` | `threshold_grid()`, `sweep()`, `select_operating_point()` |

## Tests

```bash
uv run pytest tests/ -v
```

Every file has a test file built on tiny hand-made arrays whose expected values can be checked with a pencil.