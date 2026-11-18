# HORUS AI

HORUS AI detects PTP/GNSS time-push spoofing on simulated 5G towers. It analyzes
the tower clock offset from a trusted Grandmaster and reports anomalous behavior
before the 1.5 microsecond failure limit is reached.

## Current architecture

- `src/horus_ai/data`: JSONL loading, tower alignment, and synthetic attack injection.
- `src/horus_ai/features`: window features, cross-tower peer deltas, locked/gap-free
  stretches, and training-set construction.
- `src/horus_ai/models`: the detector interface and the Isolation Forest detector.
- `src/horus_ai/training`: time-based training, validation operating-point selection,
  synthetic attack evaluation, and detector artifact creation.
- `src/horus_ai/api`: FastAPI `/health` and `/predict` endpoints plus peer-aware
  streaming detection.
- `src/horus_ai/evaluation`: metrics, debouncing, attack-event detection, lead time,
  and operating-point sweeps.
- `src/horus_ai/agent`: a standard-library telemetry agent with record and live modes.

The project is Python 3.12+, uses a `src/` layout, and is managed with
[uv](https://docs.astral.sh/uv/).

## Setup and tests

```bash
uv sync
uv run pytest
uv build
uv sync --no-editable
```

The test suite uses small hand-made arrays and does not require the large recording
files. Do not commit raw recordings or generated model artifacts.

## Running the API

```bash
uv run uvicorn horus_ai.api.main:app --reload
```

The API provides:

- `GET /health`
- `POST /predict` with one `TimingSample` JSON object

The service uses a placeholder detector when `HORUS_MODEL_PATH` is unset or does
not point to an existing artifact. A production deployment must provide and
validate a trained detector artifact.

Supported API environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `HORUS_MODEL_PATH` | Detector artifact path | Placeholder detector |
| `HORUS_WINDOW_SIZE` | Streaming window length | `30` |

## Offline training

Train the shared Isolation Forest detector from one clean JSONL recording per
tower:

```powershell
uv run python -m horus_ai.training.train `
  --output artifacts\isolation_forest.joblib
```

The four `datasets/clean/ovs*_20261115T133840Z_onehour.jsonl` recordings are
used by default. Override one or more towers with repeatable
`--tower ID=PATH` arguments. The CLI prints a Rich progress bar and a formatted
training summary.

The pipeline aligns towers, uses continuous gap-free stretches, fits only on
clean training windows, selects a threshold and debounce setting on validation
attacks, evaluates held-out sudden-jump and gradual-drift attacks, and writes
metadata beside the detector as `isolation_forest.joblib.json`. The metadata
records the feature order, window settings, tower IDs, selected operating point,
and evaluation reports.

## Telemetry agent

The agent tails a `ptp4l` log, enriches parsed samples with `pmc` context, and
either records JSONL data or sends batches to a remote endpoint:

```bash
python -m horus_ai.agent.agent `
  --config src/horus_ai/agent/config.record.example.json
```

Live mode uses `HORUS_TOKEN` for bearer authentication and can override the
configured URL with `HORUS_API_URL`:

```powershell
$env:HORUS_TOKEN = "replace-me"
python -m horus_ai.agent.agent `
  --config src/horus_ai/agent/config.live.example.json
```

The live agent currently posts to `/ingest`, while the FastAPI application
currently exposes `/predict`. This integration is unfinished and should not be
treated as a ready deployment path.

## Data and features

Recordings contain one JSONL stream per tower. Only rows with `kind=sample` and
`servo_state=s2` are measurements. Offline processing should align towers before
building cross-tower features. `find_stretches` also splits runs at timestamp gaps
and drops runs shorter than the feature window.

Each feature window contains 15 values:

- 11 features from offset, frequency, and path delay.
- 4 peer-delta features comparing the target tower with the median peer offset.

There are no real attack labels; evaluation uses synthetic sudden-jump and
gradual-drift injections.

## Project status and limitations

- The first `training/train.py` Isolation Forest pipeline is implemented, but
  its operating point still needs improvement against the protection goal.
- MLflow helpers are placeholders.
- SDN-controller alerting is not implemented.
- The topology and live streaming contract are still being finalized.
- The current model threshold is learned from the configured training quantile;
  false-alarm targets and debounce settings still require an operating-point study.

See [.github/copilot-instructions.md](.github/copilot-instructions.md) for the
full project handoff, constraints, and suggested next steps.
