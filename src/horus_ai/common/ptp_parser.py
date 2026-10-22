"""
ptp_parser.py - Horus shared ptp4l log parser.

One module, used by BOTH stages so the model sees identical rows:
  * Training:   python ptp_parser.py ovs1.log --tower ovs1 --out-dir data/
  * Deployment: the agent imports PtpLogParser and calls parse_line() on each
                new log line, then POSTs the returned row to /ingest.

Output row schema (kind == "sample"):
    ts_wall, ts_mono, tower_id, seq,
    master_offset_ns, freq_ppb, path_delay_ns,
    servo_state, gm_identity, port_state

Events (kind == "event") are non-sample lines worth keeping: port state
changes (e.g. UNCALIBRATED -> SLAVE), best-master selection, and ptp4l
"clockcheck:" warnings (clock frequency changed unexpectedly / clock jumped).
They are how you mark failovers and clock disturbances so they are not
confused with attacks, and they are a useful signal in their own right.

Only the standard library is needed.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

SAMPLE_FIELDS = [
    "ts_wall", "ts_mono", "tower_id", "seq",
    "master_offset_ns", "freq_ppb", "path_delay_ns",
    "servo_state", "gm_identity", "port_state",
]
EVENT_FIELDS = [
    "ts_wall", "ts_mono", "tower_id", "event",
    "from_state", "to_state", "gm_identity", "raw",
]

# ---------------------------------------------------------------------------
# Regular expressions
# ---------------------------------------------------------------------------

# Steady (CLOCK_MONOTONIC) timestamp, e.g. "ptp4l[1234.567]:" or "[1234.567]"
_MONO = re.compile(r"\[(?P<mono>\d+\.\d+)\]")

# Optional wall-clock prefixes (only present when logged via syslog/journal)
_ISO_TS = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
    r"(?P<tz>Z|[+-]\d{2}:?\d{2})?"
)
_SYSLOG_TS = re.compile(r"^(?P<ts>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})")

# Main sample line:
#   master offset   -16 s2 freq   +1205 path delay   5300
# Newer linuxptp may add "[domain:0]" / "[ens3]" tags before it - harmless.
# path delay is optional: some states/versions omit it.
_OFFSET = re.compile(
    r"master offset\s+(?P<offset>-?\d+)\s+(?P<state>s\d)"
    r"\s+freq\s+(?P<freq>[+-]?\d+)"
    r"(?:\s+path delay\s+(?P<delay>-?\d+))?"
)

# Events
_PORT = re.compile(
    r"port\s+\d+(?:\s*\([^)]*\))?:\s+(?P<from>[A-Z_]+)\s+to\s+(?P<to>[A-Z_]+)"
    r"\s+on\s+(?P<ev>[A-Z_]+)"
)
_BEST_MASTER = re.compile(r"selected best master clock\s+(?P<gm>[0-9a-fA-F.]+)")
# ptp4l's own sanity check, e.g. "clockcheck: clock frequency changed unexpectedly!"
_CLOCKCHECK = re.compile(r"clockcheck:\s*(?P<msg>.+)")
_LOCAL_MASTER = re.compile(r"selected local clock\s+(?P<gm>[0-9a-fA-F.]+)\s+as best master")


class PtpLogParser:
    """Stateful parser for ONE tower's ptp4l log.

    State kept between lines:
      * seq          - counter of emitted samples (lets /ingest detect gaps)
      * gm_identity  - last selected Grandmaster (attached to later samples)
      * port_state   - last port state (attached to later samples)
    The agent can also inject gm_identity / port_state from `pmc` via
    update_context().
    """

    def __init__(self, tower_id: str, default_year: Optional[int] = None):
        self.tower_id = tower_id
        self.default_year = default_year or datetime.now(timezone.utc).year
        self.seq = 0
        self.gm_identity: Optional[str] = None
        self.port_state: Optional[str] = None
        # Counters for the data-quality check
        self.stats = {"lines": 0, "samples": 0, "events": 0,
                      "ignored": 0, "malformed": 0}

    # -- public API ---------------------------------------------------------

    def update_context(self, gm_identity: Optional[str] = None,
                       port_state: Optional[str] = None) -> None:
        """Let the agent inject values read via `pmc` (e.g. every 5-10 s)."""
        if gm_identity is not None:
            self.gm_identity = gm_identity
        if port_state is not None:
            self.port_state = port_state

    def parse_line(self, line: str,
                   wall_time: Optional[float] = None) -> Optional[Dict]:
        """Parse one log line.

        Returns a dict with key "kind" == "sample" or "event", or None when
        the line is not interesting (summaries, warnings, blank lines...).

        wall_time: epoch seconds. Live agent: pass time.time(). Offline: leave
        None and the wall-clock is taken from the line prefix if it has one.
        """
        self.stats["lines"] += 1
        line = line.rstrip("\r\n")
        if not line.strip():
            self.stats["ignored"] += 1
            return None

        ts_mono = self._mono(line)
        ts_wall = wall_time if wall_time is not None else self._wall(line)

        # 1) Sample line
        m = _OFFSET.search(line)
        if m:
            self.seq += 1
            self.stats["samples"] += 1
            return {
                "kind": "sample",
                "ts_wall": ts_wall,
                "ts_mono": ts_mono,
                "tower_id": self.tower_id,
                "seq": self.seq,
                "master_offset_ns": int(m["offset"]),
                "freq_ppb": int(m["freq"]),
                "path_delay_ns": int(m["delay"]) if m["delay"] is not None else None,
                "servo_state": m["state"],
                "gm_identity": self.gm_identity,
                "port_state": self.port_state,
            }

        # A line that mentions "master offset" but did not match is malformed
        if "master offset" in line and "freq" in line:
            self.stats["malformed"] += 1
            return None

        # 2) Events
        m = _PORT.search(line)
        if m:
            self.port_state = m["to"]
            return self._event(ts_wall, ts_mono, m["ev"], m["from"], m["to"], line)

        m = _BEST_MASTER.search(line)
        if m:
            self.gm_identity = m["gm"]
            return self._event(ts_wall, ts_mono, "BEST_MASTER_SELECTED",
                               None, None, line)

        m = _LOCAL_MASTER.search(line)
        if m:
            self.gm_identity = m["gm"] + "(local)"
            return self._event(ts_wall, ts_mono, "LOCAL_CLOCK_IS_MASTER",
                               None, None, line)

        m = _CLOCKCHECK.search(line)
        if m:
            low = m["msg"].lower()
            if "frequency changed" in low:
                name = "CLOCKCHECK_FREQ_CHANGED"
            elif "jump" in low:
                name = "CLOCKCHECK_JUMP"
            else:
                name = "CLOCKCHECK_OTHER"
            return self._event(ts_wall, ts_mono, name, None, None, line)

        self.stats["ignored"] += 1
        return None

    # -- helpers ------------------------------------------------------------

    def _event(self, ts_wall, ts_mono, name, from_s, to_s, raw) -> Dict:
        self.stats["events"] += 1
        return {
            "kind": "event",
            "ts_wall": ts_wall,
            "ts_mono": ts_mono,
            "tower_id": self.tower_id,
            "event": name,
            "from_state": from_s,
            "to_state": to_s,
            "gm_identity": self.gm_identity,
            "raw": raw.strip(),
        }

    @staticmethod
    def _mono(line: str) -> Optional[float]:
        m = _MONO.search(line)
        return float(m["mono"]) if m else None

    def _wall(self, line: str) -> Optional[float]:
        """Wall-clock epoch seconds from a syslog/ISO prefix, else None.
        Naive timestamps are assumed to be UTC."""
        m = _ISO_TS.match(line)
        if m:
            ts = m["ts"].replace(" ", "T")
            fmt = "%Y-%m-%dT%H:%M:%S.%f" if "." in ts else "%Y-%m-%dT%H:%M:%S"
            dt = datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
            tz = m["tz"]
            if tz and tz != "Z":
                sign = 1 if tz[0] == "+" else -1
                digits = tz[1:].replace(":", "")
                offset = sign * (int(digits[:2]) * 3600 + int(digits[2:]) * 60)
                return dt.timestamp() - offset
            return dt.timestamp()
        m = _SYSLOG_TS.match(line)
        if m:
            try:
                dt = datetime.strptime(
                    f"{self.default_year} {' '.join(m['ts'].split())}",
                    "%Y %b %d %H:%M:%S",
                ).replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except ValueError:
                return None
        return None


# ---------------------------------------------------------------------------
# Batch helpers (training side)
# ---------------------------------------------------------------------------

def parse_lines(lines: Iterable[str], tower_id: str,
                parser: Optional[PtpLogParser] = None) -> Iterator[Dict]:
    parser = parser or PtpLogParser(tower_id)
    for line in lines:
        row = parser.parse_line(line)
        if row is not None:
            yield row


def parse_file(path: str, tower_id: str) -> Iterator[Dict]:
    with open(path, "r", errors="replace") as f:
        yield from parse_lines(f, tower_id)


def _write_csv(path: Path, fields, rows) -> int:
    n = 0
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
            n += 1
    return n


def convert(log_path: str, tower_id: str, out_dir: str) -> Dict:
    """Convert one raw log into <tower>_samples.csv and <tower>_events.csv."""
    parser = PtpLogParser(tower_id)
    samples, events = [], []
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            row = parser.parse_line(line)
            if row is None:
                continue
            (samples if row["kind"] == "sample" else events).append(row)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / f"{tower_id}_samples.csv", SAMPLE_FIELDS, samples)
    _write_csv(out / f"{tower_id}_events.csv", EVENT_FIELDS, events)
    return parser.stats


# ---------------------------------------------------------------------------
# Self-test and CLI
# ---------------------------------------------------------------------------

_SAMPLE_LOG = """\
ptp4l[100.000]: selected best master clock 964435.fffe.3cbe83
ptp4l[100.001]: port 1 (ens3): LISTENING to UNCALIBRATED on RS_SLAVE
ptp4l[101.000]: master offset     -16814 s0 freq      +0 path delay      5300
ptp4l[102.000]: master offset      -9000 s1 freq   +1200 path delay      5290
ptp4l[102.001]: port 1 (ens3): UNCALIBRATED to SLAVE on MASTER_CLOCK_SELECTED
ptp4l[103.000]: master offset        -16 s2 freq   +1205 path delay      5301
ptp4l[104.000]: [domain:0] master offset         12 s2 freq   +1210 path delay      5298
Oct 19 12:00:05 sw1 ptp4l: [105.000] master offset 7 s2 freq +1208 path delay 5300
2026-10-19T12:00:06.250+00:00 sw1 ptp4l[106.000]: master offset -3 s2 freq +1207
ptp4l[107.000]: rms 12 max 30 freq +1200 +/- 5 delay 5300 +/- 4
ptp4l[108.000]: master offset garbage s2 freq broken
ptp4l[109.000]: timed out while polling for tx timestamp
"""


def _selftest() -> None:
    p = PtpLogParser("ovs1", default_year=2026)
    out = [r for r in (p.parse_line(l) for l in _SAMPLE_LOG.splitlines()) if r]
    samples = [r for r in out if r["kind"] == "sample"]
    events = [r for r in out if r["kind"] == "event"]

    assert len(samples) == 6, len(samples)
    assert len(events) == 3, len(events)
    assert p.stats["malformed"] == 1
    assert samples[0]["master_offset_ns"] == -16814
    assert samples[0]["servo_state"] == "s0"
    assert samples[0]["gm_identity"] == "964435.fffe.3cbe83"
    assert samples[0]["port_state"] == "UNCALIBRATED"
    assert samples[2]["port_state"] == "SLAVE"          # after the event
    assert [s["seq"] for s in samples] == list(range(1, 7))
    assert samples[3]["ts_mono"] == 104.0               # [domain:0] form
    assert samples[4]["ts_wall"] is not None            # syslog prefix
    assert samples[5]["ts_wall"] is not None            # ISO prefix
    assert samples[5]["path_delay_ns"] is None          # optional field
    # Real lines copied from the network team's test screenshot
    real = PtpLogParser("ovs1")
    real_lines = [
        "ptp4l[12078.188]: clockcheck: clock frequency changed unexpectedly!",
        "ptp4l[12078.189]: master offset   -4014654 s2 freq +3303740 path delay   4757235",
        "ptp4l[12079.191]: clockcheck: clock frequency changed unexpectedly!",
        "ptp4l[12079.191]: master offset    -972835 s2 freq +3606949 path delay   4757235",
    ]
    rr = [r for r in (real.parse_line(l) for l in real_lines) if r]
    assert [r["kind"] for r in rr] == ["event", "sample", "event", "sample"]
    assert rr[0]["event"] == "CLOCKCHECK_FREQ_CHANGED"
    assert rr[1]["master_offset_ns"] == -4014654 and rr[1]["freq_ppb"] == 3303740
    assert rr[1]["path_delay_ns"] == 4757235 and rr[1]["servo_state"] == "s2"
    assert real.stats["malformed"] == 0

    p.update_context(gm_identity="abc.def", port_state="SLAVE")
    live = p.parse_line("ptp4l[110.0]: master offset 1 s2 freq +5 path delay 9",
                        wall_time=1.0)
    assert live["gm_identity"] == "abc.def" and live["ts_wall"] == 1.0
    print("self-test passed:", p.stats)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Convert a ptp4l log to CSV.")
    ap.add_argument("log", nargs="?", help="path to the raw ptp4l log")
    ap.add_argument("--tower", help="tower id, e.g. ovs1")
    ap.add_argument("--out-dir", default="parsed")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        _selftest()
        return 0
    if not args.log or not args.tower:
        ap.error("log path and --tower are required (or use --selftest)")

    stats = convert(args.log, args.tower, args.out_dir)
    print(f"{args.tower}: {stats}")
    if stats["malformed"]:
        print("WARNING: malformed 'master offset' lines found - check the "
              "log format against the regex.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())