"""
agent.py - Horus telemetry agent (runs on each OVS "tower").

ONE agent, TWO modes (set "mode" in the config file, or pass --mode):

  mode "record"  -> TRAINING. Saves every row to local .jsonl files. No API,
                    no token needed. Copy the files off the switch afterwards.
  mode "live"    -> DEPLOYMENT. Sends every row to <api_url>/ingest with a
                    bearer token, retries if the API is down, sends heartbeats.

Both modes share the exact same steps before the output:
  1. Tail the ptp4l log file (survives log rotation and partial lines).
  2. Parse each line with the shared parser (horus_ai.common.ptp_parser).
  3. Every few seconds, ask `pmc` for the Grandmaster ID and port state and
     feed them to the parser as context.
  4. Batch the rows in a bounded in-memory queue.
Only the last step differs: write to a file (record) or POST to the API (live).

Standard library only (it runs on the switches - no pandas/torch here).

Usage:
    # training
    python -m horus_ai.agent.agent --config config.record.example.json
    # deployment
    export HORUS_TOKEN=...            # never put the token in the config file
    python -m horus_ai.agent.agent --config config.live.example.json
    # just look at the parsed rows
    python -m horus_ai.agent.agent --config agent.json --dry-run --from-start
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Deque, Dict, Iterator, List, Optional

try:  # normal case: repo layout / installed package
    from horus_ai.common.ptp_parser import PtpLogParser
except ImportError:  # fallback: ptp_parser.py copied next to this file
    from ptp_parser import PtpLogParser  # type: ignore

VERSION = "0.1.0"
log = logging.getLogger("horus_ai.agent")

DEFAULTS: Dict = {
    "mode": "live",                # "record" (training) or "live" (deployment)
    "tower_id": None,              # required, e.g. "ovs1"
    "api_url": None,               # required in live mode, e.g. "http://10.0.20.5:8000"
    "output_dir": "/var/lib/horus",  # record mode: where .jsonl files are written
    "output_max_mb": 100,          # record mode: start a new file after this size
    "log_path": "/var/log/ptp4l.log",
    "from_start": False,           # False: only new lines (live). True: replay file first
    "flush_interval": 1.0,         # seconds between batches
    "batch_max": 50,               # max rows per POST
    "heartbeat_interval": 5.0,     # send an empty batch if nothing sent for this long
    "queue_max": 20000,            # ~40 min of buffer at 8 samples/s
    "http_timeout": 3.0,
    "max_backoff": 30.0,
    "pmc_enabled": True,
    "pmc_cmd": "pmc -u -b 0",
    "pmc_interval": 10.0,
    "stats_interval": 60.0,
}

_PMC_GM = re.compile(r"gmIdentity\s+(\S+)")
_PMC_PORT = re.compile(r"portState\s+(\S+)")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: Optional[str], overrides: Optional[Dict] = None) -> Dict:
    cfg = dict(DEFAULTS)
    if path:
        with open(path) as f:
            cfg.update(json.load(f))
    if os.environ.get("HORUS_API_URL"):
        cfg["api_url"] = os.environ["HORUS_API_URL"]
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    if cfg["mode"] not in ("record", "live"):
        raise ValueError("config: 'mode' must be \"record\" or \"live\"")
    required = ["tower_id"] + (["api_url"] if cfg["mode"] == "live" else ["output_dir"])
    for key in required:
        if not cfg.get(key):
            raise ValueError(f"config: '{key}' is required in {cfg['mode']} mode")
    if cfg.get("api_url"):
        cfg["api_url"] = cfg["api_url"].rstrip("/")
    return cfg


# ---------------------------------------------------------------------------
# Log tailing (like `tail -F`)
# ---------------------------------------------------------------------------

def follow(path: str, stop: threading.Event, from_start: bool = False,
           poll: float = 0.2) -> Iterator[str]:
    """Yield complete lines as they are appended to `path`.

    Handles: file not existing yet, log rotation (new inode), truncation,
    and half-written last lines (waits for the newline).
    """
    f = None
    inode = None
    first_open = True
    while not stop.is_set():
        if f is None:
            try:
                f = open(path, "rb")
            except FileNotFoundError:
                stop.wait(1.0)
                continue
            inode = os.fstat(f.fileno()).st_ino
            if first_open and not from_start:
                f.seek(0, os.SEEK_END)
            first_open = False

        pos = f.tell()
        raw = f.readline()
        if raw:
            if raw.endswith(b"\n"):
                yield raw.decode("utf-8", errors="replace")
            else:  # partial line: rewind and wait for the rest
                f.seek(pos)
                stop.wait(poll)
            continue

        # No new data: check for rotation / truncation, then wait.
        try:
            st = os.stat(path)
            if st.st_ino != inode or st.st_size < f.tell():
                log.info("log rotated or truncated, reopening %s", path)
                f.close()
                f = None
                continue
        except FileNotFoundError:
            pass
        stop.wait(poll)

    if f is not None:
        f.close()


# ---------------------------------------------------------------------------
# pmc helper
# ---------------------------------------------------------------------------

def run_pmc(base_cmd: List[str], command: str, timeout: float = 2.0) -> Optional[str]:
    try:
        res = subprocess.run(base_cmd + [command], capture_output=True,
                             text=True, timeout=timeout)
        return res.stdout
    except FileNotFoundError:
        raise
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.debug("pmc '%s' failed: %s", command, exc)
        return None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self, cfg: Dict, token: Optional[str] = None,
                 dry_run: bool = False, out=None):
        self.cfg = cfg
        self.token = token
        self.dry_run = dry_run
        self.out = out or sys.stdout

        # One id per agent start. /ingest uses it to tell a restart
        # (seq goes back to 1) apart from lost samples.
        self.session_id = uuid.uuid4().hex[:12]

        self.parser = PtpLogParser(cfg["tower_id"])
        self._parser_lock = threading.Lock()

        self._q: Deque[Dict] = collections.deque()
        self._q_lock = threading.Lock()

        self.stop_event = threading.Event()
        self._threads: List[threading.Thread] = []

        self.dropped = 0
        self.sent = 0
        self.post_failures = 0

        # record-mode file state
        self._started = time.time()
        self._out_file = None
        self._out_bytes = 0
        self._out_part = 0

    # -- queue ---------------------------------------------------------------

    def _enqueue(self, row: Dict) -> None:
        with self._q_lock:
            if len(self._q) >= self.cfg["queue_max"]:
                self._q.popleft()          # drop the OLDEST row, count it
                self.dropped += 1
            self._q.append(row)

    def _take(self, n: int) -> List[Dict]:
        with self._q_lock:
            return [self._q.popleft() for _ in range(min(n, len(self._q)))]

    def _putback(self, batch: List[Dict]) -> None:
        with self._q_lock:
            room = self.cfg["queue_max"] - len(self._q)
            excess = max(0, len(batch) - room)
            if excess:
                self.dropped += excess
                batch = batch[excess:]
            self._q.extendleft(reversed(batch))

    def queue_len(self) -> int:
        with self._q_lock:
            return len(self._q)

    # -- network -------------------------------------------------------------

    def _payload(self, batch: List[Dict]) -> Dict:
        samples, events = [], []
        for row in batch:
            row = dict(row)
            kind = row.pop("kind")
            (samples if kind == "sample" else events).append(row)
        return {
            "tower_id": self.cfg["tower_id"],
            "session_id": self.session_id,
            "agent_version": VERSION,
            "agent_ts": time.time(),
            "dropped_total": self.dropped,
            "samples": samples,
            "events": events,
        }

    def _open_next_file(self) -> None:
        if self._out_file is not None:
            self._out_file.close()
        os.makedirs(self.cfg["output_dir"], exist_ok=True)
        self._out_part += 1
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(self._started))
        name = f"{self.cfg['tower_id']}_{stamp}_{self.session_id}_p{self._out_part:03d}.jsonl"
        path = os.path.join(self.cfg["output_dir"], name)
        self._out_file = open(path, "a", encoding="utf-8")
        self._out_bytes = 0
        log.info("recording to %s", path)

    def _write_records(self, batch: List[Dict]) -> bool:
        """Record mode: append rows (samples AND events, with 'kind') as JSON lines."""
        try:
            max_bytes = self.cfg["output_max_mb"] * 1024 * 1024
            if self._out_file is None or self._out_bytes >= max_bytes:
                self._open_next_file()
            data = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in batch)
            self._out_file.write(data)
            self._out_file.flush()
            self._out_bytes += len(data.encode("utf-8"))
            return True
        except OSError as exc:
            log.error("cannot write record file: %s", exc)
            return False

    def _post(self, batch: List[Dict]) -> bool:
        """Deliver one batch: print (dry run), write to file (record) or POST (live)."""
        if self.dry_run:
            for row in batch:
                print(json.dumps(row), file=self.out, flush=True)
            return True
        if self.cfg["mode"] == "record":
            return self._write_records(batch)

        payload = self._payload(batch)

        req = urllib.request.Request(
            self.cfg["api_url"] + "/ingest",
            data=json.dumps(payload).encode(),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.token}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.cfg["http_timeout"]) as r:
                return 200 <= r.status < 300
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                log.error("API rejected the token (HTTP %s) - check HORUS_TOKEN", exc.code)
            else:
                log.warning("API returned HTTP %s", exc.code)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            log.warning("API unreachable: %s", exc)
        return False

    # -- threads -------------------------------------------------------------

    def _reader_loop(self) -> None:
        log.info("tailing %s (from_start=%s)", self.cfg["log_path"], self.cfg["from_start"])
        for line in follow(self.cfg["log_path"], self.stop_event,
                           from_start=self.cfg["from_start"]):
            with self._parser_lock:
                row = self.parser.parse_line(line, wall_time=time.time())
            if row is not None:
                self._enqueue(row)

    def _sender_loop(self) -> None:
        cfg = self.cfg
        flush = cfg["flush_interval"]
        wait = flush
        backoff = flush
        last_post = 0.0
        last_stats = time.monotonic()
        failing = False
        # Heartbeats only make sense when talking to the API
        heartbeat = cfg["heartbeat_interval"] if cfg["mode"] == "live" else float("inf")

        while not self.stop_event.is_set():
            if self.stop_event.wait(wait):
                break
            batch = self._take(cfg["batch_max"])
            now = time.monotonic()

            if not batch and now - last_post < heartbeat:
                wait = flush
            elif self._post(batch):
                if failing:
                    log.info("delivery recovered")
                    failing = False
                last_post = now
                self.sent += len(batch)
                backoff = flush
                wait = 0.05 if self.queue_len() else flush   # drain backlog fast
            else:
                self._putback(batch)
                self.post_failures += 1
                failing = True
                backoff = min(max(backoff * 2, flush), cfg["max_backoff"])
                wait = backoff

            if now - last_stats >= cfg["stats_interval"]:
                last_stats = now
                log.info("stats: sent=%d queued=%d dropped=%d failures=%d",
                         self.sent, self.queue_len(), self.dropped, self.post_failures)

        # Final flush on shutdown (best effort, max 5 s)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            batch = self._take(cfg["batch_max"])
            if not batch:
                break
            if not self._post(batch):
                self._putback(batch)
                break
            self.sent += len(batch)

    def _pmc_loop(self) -> None:
        base = shlex.split(self.cfg["pmc_cmd"])
        while not self.stop_event.wait(self.cfg["pmc_interval"]):
            try:
                gm_out = run_pmc(base, "GET TIME_STATUS_NP")
                port_out = run_pmc(base, "GET PORT_DATA_SET")
            except FileNotFoundError:
                log.warning("pmc not found - disabling pmc context polling")
                return
            gm = _PMC_GM.search(gm_out or "")
            port = _PMC_PORT.search(port_out or "")
            if gm or port:
                with self._parser_lock:
                    self.parser.update_context(
                        gm_identity=gm.group(1) if gm else None,
                        port_state=port.group(1) if port else None,
                    )

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        targets = [self._reader_loop, self._sender_loop]
        if self.cfg["pmc_enabled"] and not self.dry_run:
            targets.append(self._pmc_loop)
        for t in targets:
            th = threading.Thread(target=t, name=t.__name__, daemon=True)
            th.start()
            self._threads.append(th)
        dest = ("(dry run)" if self.dry_run else
                self.cfg["output_dir"] if self.cfg["mode"] == "record" else self.cfg["api_url"])
        log.info("agent %s started: mode=%s tower=%s session=%s -> %s",
                 VERSION, self.cfg["mode"], self.cfg["tower_id"], self.session_id, dest)

    def stop(self, timeout: float = 8.0) -> None:
        self.stop_event.set()
        for th in self._threads:
            th.join(timeout)
        if self._out_file is not None:
            self._out_file.close()
            self._out_file = None
        log.info("agent stopped: sent=%d dropped=%d", self.sent, self.dropped)

    def run_forever(self) -> None:
        def _handler(signum, _frame):
            log.info("signal %s received, shutting down", signum)
            self.stop_event.set()

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
        self.start()
        while not self.stop_event.wait(0.5):
            pass
        self.stop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Horus telemetry agent")
    ap.add_argument("--config", help="path to agent JSON config")
    ap.add_argument("--mode", choices=["record", "live"],
                    help="override the mode from the config file")
    ap.add_argument("--dry-run", action="store_true",
                    help="print parsed rows to stdout instead of POSTing")
    ap.add_argument("--from-start", action="store_true",
                    help="read the existing log from the beginning, then keep following")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        cfg = load_config(args.config, {"mode": args.mode,
                                        "from_start": True if args.from_start else None})
    except (ValueError, OSError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    token = os.environ.get("HORUS_TOKEN")
    if cfg["mode"] == "live" and not token and not args.dry_run:
        print("HORUS_TOKEN environment variable is required in live mode", file=sys.stderr)
        return 2

    Agent(cfg, token=token, dry_run=args.dry_run).run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
