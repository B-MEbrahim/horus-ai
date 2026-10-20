"""End-to-end test: fake API (fails twice first) + real agent + growing log file.

Run from the repo root:  PYTHONPATH=src python tests/test_agent.py
"""
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from horus_ai.agent.agent import Agent, DEFAULTS  # noqa: E402

received = []          # parsed JSON bodies
auth_seen = set()
fail_first = {"n": 2}  # first 2 POSTs get HTTP 503


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        auth_seen.add(self.headers.get("Authorization"))
        if fail_first["n"] > 0:
            fail_first["n"] -= 1
            self.send_response(503)
            self.end_headers()
            return
        received.append(json.loads(body))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *a):
        pass


def main():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    tmp = tempfile.mkdtemp()
    logpath = os.path.join(tmp, "ptp4l.log")
    open(logpath, "w").close()

    cfg = dict(DEFAULTS, tower_id="ovs1",
               api_url=f"http://127.0.0.1:{srv.server_port}",
               log_path=logpath, flush_interval=0.1, heartbeat_interval=0.3,
               max_backoff=0.3, pmc_enabled=False, stats_interval=999)
    agent = Agent(cfg, token="secret")
    agent.start()
    time.sleep(0.4)

    # Write 20 samples, the last line deliberately in two pieces (partial line)
    with open(logpath, "a") as f:
        f.write("ptp4l[1.0]: port 1 (ens3): UNCALIBRATED to SLAVE on MASTER_CLOCK_SELECTED\n")
        for i in range(1, 20):
            f.write(f"ptp4l[{i + 1}.0]: master offset {i} s2 freq +{i} path delay 100\n")
            f.flush()
        f.write("ptp4l[21.0]: master offset 20 s2 fr")
        f.flush()
        time.sleep(0.3)
        f.write("eq +20 path delay 100\n")

    time.sleep(1.5)
    # simulate log rotation: new file, new inode
    os.rename(logpath, logpath + ".1")
    with open(logpath, "w") as f:
        f.write("ptp4l[30.0]: master offset 99 s2 freq +9 path delay 100\n")
    time.sleep(1.0)
    agent.stop()
    srv.shutdown()

    samples = [s for b in received for s in b["samples"]]
    events = [e for b in received for e in b["events"]]
    seqs = [s["seq"] for s in samples]

    assert auth_seen == {"Bearer secret"}, auth_seen
    assert agent.post_failures >= 2, agent.post_failures           # retried after 503s
    assert seqs == list(range(1, 22)), seqs                       # all 21, in order, no dupes
    assert len(events) == 1 and events[0]["event"] == "MASTER_CLOCK_SELECTED"
    assert samples[19]["master_offset_ns"] == 20                  # partial line joined
    assert samples[-1]["master_offset_ns"] == 99                  # after rotation
    assert all("kind" not in s for s in samples)
    assert len({b["session_id"] for b in received}) == 1
    hb = [b for b in received if not b["samples"] and not b["events"]]
    assert hb, "expected at least one heartbeat"
    assert agent.dropped == 0
    print(f"OK: {len(samples)} samples, {len(events)} event, "
          f"{len(hb)} heartbeats, {agent.post_failures} failed POSTs retried")


def test_record_mode():
    """Record mode: no API at all, rows land in .jsonl files, files rotate by size."""
    import glob
    tmp = tempfile.mkdtemp()
    logpath = os.path.join(tmp, "ptp4l.log")
    outdir = os.path.join(tmp, "out")
    open(logpath, "w").close()

    cfg = dict(DEFAULTS, mode="record", tower_id="ovs2", log_path=logpath,
               output_dir=outdir, output_max_mb=0.002,       # ~2 KB -> forces rotation
               flush_interval=0.1, pmc_enabled=False, stats_interval=999)
    agent = Agent(cfg)                                       # no token, no api_url
    agent.start()
    time.sleep(0.3)
    with open(logpath, "a") as f:
        f.write("ptp4l[1.0]: port 1 (ens3): UNCALIBRATED to SLAVE on MASTER_CLOCK_SELECTED\n")
        for i in range(1, 41):
            f.write(f"ptp4l[{i + 1}.0]: master offset {i} s2 freq +{i} path delay 100\n")
            f.flush()
            time.sleep(0.02)
    time.sleep(0.8)
    agent.stop()

    files = sorted(glob.glob(os.path.join(outdir, "ovs2_*.jsonl")))
    rows = [json.loads(l) for fn in files for l in open(fn)]
    samples = [r for r in rows if r["kind"] == "sample"]
    events = [r for r in rows if r["kind"] == "event"]
    assert len(files) >= 2, files                            # rotated
    assert [r["seq"] for r in samples] == list(range(1, 41))
    assert len(events) == 1
    assert all(r["ts_wall"] is not None for r in samples)    # wall-clock stamped by the agent
    print(f"OK record mode: {len(samples)} samples in {len(files)} files")


if __name__ == "__main__":
    main()
    test_record_mode()
