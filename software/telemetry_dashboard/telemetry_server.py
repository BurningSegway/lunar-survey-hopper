#!/usr/bin/env python3
"""Serial telemetry -> web dashboard bridge (Teleplot-style).

Reads lines from a Teensy over USB serial, parses Teleplot formatted values
(``>name:value``) and streams them to any number of browsers via
Server-Sent Events.  Lines that are not telemetry (``Starting ODriveCAN demo``,
warnings, ...) are shown in the dashboard's log console.

If the firmware uses include/telemetry.h, the dashboard can also switch the
Teensy's print channels on/off and change the print rate.  The chosen channels
are remembered and re-applied when the Teensy resets.

Only dependency besides the standard library: pyserial.

    python3 telemetry_server.py                  # auto-detect Teensy, port 8080
    python3 telemetry_server.py --port /dev/ttyACM0 --http-port 8000
    python3 telemetry_server.py --demo           # fake data, no hardware needed

Then open http://<pi-address>:8080 in a browser.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import queue
import random
import re
import sys
import threading
import time
from collections import deque
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    import serial
    import serial.tools.list_ports
except ImportError:  # allow --demo without pyserial installed
    serial = None

HERE = Path(__file__).resolve().parent
INDEX_HTML = HERE / "static" / "index.html"

TEENSY_USB_VID = 0x16C0
CATALOG_PREFIX = "#tlm "
CHANNEL_NAME_RE = re.compile(r"^[A-Za-z0-9_]{1,32}$")


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_line(line: str):
    """Parse one serial line.

    Returns one of
        ("data", [(name, value), ...])   Teleplot numeric telemetry
        ("catalog", {...})               channel catalog from telemetry.h
        ("log", text)                    anything else
    Supported Teleplot forms:
        >name:1.23
        >name:1.23§unit|flags
        >name:<timestamp>:1.23          (device timestamp is ignored)
        >name:ts:1;ts:2                 (several points)
    Text telemetry (>name:text|t) and anything unparseable goes to the log.
    """
    if line.startswith(CATALOG_PREFIX):
        try:
            cat = json.loads(line[len(CATALOG_PREFIX):])
            if isinstance(cat, dict) and isinstance(cat.get("ch"), list):
                return "catalog", cat
        except ValueError:
            pass
        return "log", line
    if not line.startswith(">") or ":" not in line:
        return "log", line
    body = line[1:].split("|", 1)[0].split("§", 1)[0]
    name, rest = body.split(":", 1)
    name = name.strip()
    if not name:
        return "log", line
    out = []
    for part in rest.split(";"):
        val = part.rsplit(":", 1)[-1].strip()
        try:
            f = float(val)
        except ValueError:
            return "log", f"{name}: {rest.strip()}"
        if math.isfinite(f):
            out.append((name, f))
    return ("data", out) if out else ("log", line)


# --------------------------------------------------------------------------- #
# Hub: keeps history, batches updates, fans out to clients
# --------------------------------------------------------------------------- #
class Hub:
    def __init__(self, history_s: float, max_points: int, recorder: Recorder):
        self.history_s = history_s
        self.max_points = max_points
        self.recorder = recorder
        self.lock = threading.Lock()
        self.series: dict[str, deque] = {}
        self.logs: deque = deque(maxlen=500)
        self.pending_series: dict[str, list] = {}
        self.pending_logs: list = []
        self.clients: set[queue.Queue] = set()
        self.status = {"serial": "starting", "port": None, "recording": None}
        self.status_dirty = True
        self.catalog: dict | None = None
        self.catalog_dirty = False
        self.on_catalog = None  # callback(catalog) set by the device link
        self._rate_count = 0
        self._rate_t0 = time.time()
        self.rate = 0.0

    # -- producers ---------------------------------------------------------- #
    def add_line(self, line: str, t: float | None = None):
        t = time.time() if t is None else t
        kind, payload = parse_line(line)
        with self.lock:
            self._rate_count += 1
            if kind == "data":
                for name, v in payload:
                    dq = self.series.get(name)
                    if dq is None:
                        dq = self.series[name] = deque(maxlen=self.max_points)
                    dq.append((t, v))
                    p = self.pending_series.setdefault(name, [[], []])
                    p[0].append(round(t, 4))
                    p[1].append(v)
            elif kind == "catalog":
                self.catalog = payload
                self.catalog_dirty = True
            else:
                self.logs.append((t, payload))
                self.pending_logs.append([round(t, 4), payload])
        if kind == "catalog" and self.on_catalog:
            self.on_catalog(payload)
        self.recorder.write(t, kind, line if kind == "catalog" else payload)

    def set_status(self, **kw):
        with self.lock:
            self.status.update(kw)
            self.status_dirty = True
            if kw.get("serial") != "connected" and "serial" in kw:
                self.catalog, self.catalog_dirty = None, True
        if "serial" in kw:
            msg = f"[bridge] serial {kw['serial']}" + (f" ({kw.get('port')})" if kw.get("port") else "")
            self.add_line(msg)

    def set_recording(self, on: bool):
        if on and not self.recorder.active:
            path = self.recorder.start()
            self.set_status(recording=path.name)
            self.add_line(f"[bridge] recording started: {path}")
            if self.catalog:  # note which channels were on at the start of the file
                self.recorder.write(time.time(), "log", CATALOG_PREFIX + json.dumps(self.catalog))
        elif not on and self.recorder.active:
            path = self.recorder.path
            self.add_line(f"[bridge] recording stopped: {path}")
            self.recorder.stop()
            self.set_status(recording=None)

    def clear(self):
        with self.lock:
            self.series.clear()
            self.logs.clear()
            self.pending_series.clear()
            self.pending_logs.clear()
        self._broadcast(json.dumps({"type": "clear"}))

    # -- consumers ---------------------------------------------------------- #
    def snapshot(self) -> str:
        now = time.time()
        cutoff = now - self.history_s
        with self.lock:
            series = {}
            for name, dq in self.series.items():
                pts = [p for p in dq if p[0] >= cutoff]
                series[name] = [[round(p[0], 4) for p in pts], [p[1] for p in pts]]
            logs = [[round(t, 4), s] for t, s in list(self.logs)[-200:]]
            status = dict(self.status, rate=self.rate)
            catalog = self.catalog
        return json.dumps({"type": "snapshot", "now": now, "history": self.history_s,
                           "series": series, "logs": logs, "status": status, "tlm": catalog})

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=400)
        with self.lock:
            self.clients.add(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self.lock:
            self.clients.discard(q)

    def _broadcast(self, msg: str):
        with self.lock:
            clients = list(self.clients)
        for q in clients:
            try:
                q.put_nowait(msg)
            except queue.Full:  # slow client: drop it, browser will reconnect
                self.unsubscribe(q)

    def flush_loop(self, period: float = 0.05):
        """Push batched updates to clients ~20x per second."""
        while True:
            time.sleep(period)
            now = time.time()
            with self.lock:
                if now - self._rate_t0 >= 1.0:
                    self.rate = self._rate_count / (now - self._rate_t0)
                    self._rate_count, self._rate_t0 = 0, now
                    self.status_dirty = True
                if not (self.pending_series or self.pending_logs or self.status_dirty or self.catalog_dirty):
                    continue
                msg = {"type": "batch", "now": now, "series": self.pending_series, "logs": self.pending_logs}
                if self.status_dirty:
                    msg["status"] = dict(self.status, rate=round(self.rate, 1))
                    self.status_dirty = False
                if self.catalog_dirty:
                    msg["tlm"] = self.catalog
                    self.catalog_dirty = False
                self.pending_series, self.pending_logs = {}, []
            self._broadcast(json.dumps(msg))

    def export_csv(self) -> str:
        """Wide CSV of the server history (one row per millisecond, like Teleplot)."""
        with self.lock:
            names = sorted(self.series)
            rows: dict[float, dict] = {}
            for n in names:
                for t, v in self.series[n]:
                    rows.setdefault(round(t, 3), {})[n] = v
        lines = ["timestamp_s," + ",".join(names)]
        for t in sorted(rows):
            r = rows[t]
            lines.append(f"{t:.3f}," + ",".join("" if n not in r else repr(r[n]) for n in names))
        return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Recording to disk
# --------------------------------------------------------------------------- #
class Recorder:
    """Long-format CSV: timestamp_s,name,value  (log lines use name='log').

    Off until start() is called (Record button in the dashboard or --record).
    Every start() opens a new file in the recording directory."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.path: Path | None = None
        self._f = None
        self._w = None
        self._lock = threading.Lock()
        self._last_flush = 0.0

    @property
    def active(self) -> bool:
        return self._f is not None

    def start(self) -> Path:
        with self._lock:
            if self._f is None:
                self.directory.mkdir(parents=True, exist_ok=True)
                self.path = self.directory / f"telemetry_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
                self._f = open(self.path, "w", newline="", encoding="utf-8")
                self._w = csv.writer(self._f)
                self._w.writerow(["timestamp_s", "name", "value"])
            return self.path

    def stop(self):
        with self._lock:
            if self._f is not None:
                self._f.close()
                self._f = self._w = None

    def write(self, t: float, kind: str, payload):
        with self._lock:
            if self._w is None:
                return
            if kind == "data":
                for name, v in payload:
                    self._w.writerow([f"{t:.4f}", name, v])
            else:
                self._w.writerow([f"{t:.4f}", "log", payload])
            if t - self._last_flush > 1.0:
                self._f.flush()
                self._last_flush = t


# --------------------------------------------------------------------------- #
# Telemetry channel control (talks to include/telemetry.h on the Teensy)
# --------------------------------------------------------------------------- #
class ChannelControl:
    """Remembers the channel selection made in the dashboard and re-applies it
    whenever the Teensy (re)boots or the serial link reconnects."""

    def __init__(self, hub: Hub, state_file: Path):
        self.hub = hub
        self.state_file = state_file
        self.link = None  # object with send(text) -> bool
        self.lock = threading.Lock()
        self.desired: dict = {"channels": {}, "rate": None}
        self.needs_apply = True
        try:
            self.desired.update(json.loads(state_file.read_text()))
        except (OSError, ValueError):
            pass
        hub.on_catalog = self._on_catalog

    def _save(self):
        try:
            self.state_file.write_text(json.dumps(self.desired, indent=1))
        except OSError as e:
            print(f"could not save {self.state_file}: {e}", file=sys.stderr)

    def connected(self):
        """Call after the serial port (re)opens: ask for the catalog, re-apply on reply."""
        with self.lock:
            self.needs_apply = True
        self.query()

    def query(self):
        # The leading empty line terminates any partial line already sitting in the
        # Teensy's command buffer (e.g. tty echo of its own output while the port was
        # opening), so "tlm?" always arrives as a clean line.
        self.send("")
        self.send("tlm?")

    def _on_catalog(self, cat: dict):
        with self.lock:
            if not (self.needs_apply or cat.get("boot")):
                return
            self.needs_apply = False
            cmds = []
            for ch in cat.get("ch", []):
                want = self.desired["channels"].get(ch.get("n"))
                if want is not None and bool(want) != bool(ch.get("on")):
                    cmds.append(f"tlm {ch['n']} {1 if want else 0}")
            rate = self.desired.get("rate")
            if rate and rate != cat.get("rate"):
                cmds.append(f"tlm rate {int(rate)}")
        if cmds:
            self.hub.add_line(f"[bridge] re-applying channel selection: {', '.join(c[4:] for c in cmds)}")
        for c in cmds:
            self.send(c)

    def set_channel(self, name: str, on: bool) -> bool:
        if not CHANNEL_NAME_RE.match(name):
            return False
        with self.lock:
            if name == "all":
                cat = self.hub.catalog or {"ch": []}
                for ch in cat["ch"]:
                    self.desired["channels"][ch["n"]] = on
            else:
                self.desired["channels"][name] = on
            self._save()
        return self.send(f"tlm {name} {1 if on else 0}")

    def set_rate(self, ms: int) -> bool:
        ms = max(10, min(1000, int(ms)))
        with self.lock:
            self.desired["rate"] = ms
            self._save()
        return self.send(f"tlm rate {ms}")

    def send(self, text: str) -> bool:
        link = self.link
        return bool(link and link.send(text))


# --------------------------------------------------------------------------- #
# Serial / demo sources
# --------------------------------------------------------------------------- #
def find_port() -> str | None:
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if p.vid == TEENSY_USB_VID:
            return p.device
    for pattern in ("/dev/ttyACM*", "/dev/ttyUSB*"):
        found = sorted(glob.glob(pattern))
        if found:
            return found[0]
    return None


class SerialLink:
    """Reads forever; reconnects automatically when the Teensy resets/unplugs."""

    def __init__(self, hub: Hub, control: ChannelControl, port: str | None, baud: int):
        self.hub, self.control, self.port_arg, self.baud = hub, control, port, baud
        self.ser = None
        self.write_lock = threading.Lock()
        control.link = self

    def send(self, text: str) -> bool:
        with self.write_lock:
            ser = self.ser
            if ser is None:
                return False
            try:
                ser.write((text.strip() + "\n").encode())
                return True
            except (serial.SerialException, OSError):
                return False

    def run(self):
        waiting_logged = False
        while True:
            port = self.port_arg or find_port()
            if not port:
                if not waiting_logged:
                    self.hub.set_status(serial="waiting for device", port=None)
                    waiting_logged = True
                time.sleep(1.0)
                continue
            try:
                with serial.Serial(port, self.baud, timeout=1.0, write_timeout=0.5) as ser:
                    waiting_logged = False
                    self.ser = ser
                    self.hub.set_status(serial="connected", port=port)
                    time.sleep(0.1)          # let the port settle, drop what arrived meanwhile
                    ser.reset_input_buffer()
                    self.control.connected()
                    queried_at, queries = time.time(), 1
                    buf = b""
                    while True:
                        # No channel list yet (reply lost, Teensy still in setup, ...): ask again.
                        if self.hub.catalog is None and queries < 15 and time.time() - queried_at > 2.0:
                            self.control.query()
                            queried_at, queries = time.time(), queries + 1
                        chunk = ser.read(ser.in_waiting or 1)
                        if not chunk:
                            continue
                        t = time.time()
                        buf += chunk
                        *lines, buf = buf.split(b"\n")
                        for raw in lines:
                            line = raw.decode("utf-8", errors="replace").strip()
                            if line:
                                self.hub.add_line(line, t)
                        if len(buf) > 4096:  # garbage without newlines
                            buf = b""
            except (serial.SerialException, OSError) as e:
                with self.write_lock:
                    self.ser = None
                self.hub.set_status(serial=f"disconnected: {e.__class__.__name__}", port=port)
                waiting_logged = False
                time.sleep(1.0)


class DemoLink:
    """Emulates the LQR firmware (including telemetry.h commands) at 50 Hz."""

    CHANNELS = [
        ["imu_age", "QuatAgeMs GyroAgeMs", True],
        ["lqr_torque", "LQR-tau-x LQR-tau-y LQR-tau-z", True],
        ["lqr_state", "qe-x qe-y qe-z omega-x omega-y omega-z", False],
        ["flight", "AccelerationMagnitude Airborne", True],
        ["euler", "Roll Pitch Yaw", False],
    ]

    def __init__(self, hub: Hub, control: ChannelControl):
        self.hub, self.control = hub, control
        self.rate_ms = 20
        self.cmds: queue.Queue = queue.Queue()
        control.link = self

    def send(self, text: str) -> bool:
        self.cmds.put(text.strip())
        return True

    def catalog(self, boot=False):
        cat = {"rate": self.rate_ms, "ch": [{"n": n, "on": int(on), "v": v} for n, v, on in self.CHANNELS]}
        if boot:
            cat["boot"] = 1
        return CATALOG_PREFIX + json.dumps(cat, separators=(",", ":"))

    def handle(self, cmd: str):
        parts = cmd.split()
        if not parts:
            return
        if parts[0] == "tlm?" or parts == ["tlm"] or parts[:2] == ["tlm", "?"]:
            self.hub.add_line(self.catalog())
            return
        if parts[0] != "tlm" or len(parts) < 3:
            self.hub.add_line(f"Unknown command: {parts[0]}")
            return
        target, val = parts[1], int(parts[2]) if parts[2].lstrip("-").isdigit() else 0
        if target == "rate":
            self.rate_ms = max(10, min(1000, val))
        elif target == "all":
            for ch in self.CHANNELS:
                ch[2] = bool(val)
        else:
            ch = next((c for c in self.CHANNELS if c[0] == target), None)
            if not ch:
                self.hub.add_line(f"tlm: unknown channel {target}")
                return
            ch[2] = bool(val)
        self.hub.add_line(self.catalog())

    def run(self):
        hub = self.hub
        hub.set_status(serial="demo mode", port="demo")
        hub.add_line("Starting ODriveCAN demo")
        hub.add_line(self.catalog(boot=True))
        hub.add_line("ODrives running!")
        t0 = time.time()
        airborne = 0
        while True:
            while not self.cmds.empty():
                self.handle(self.cmds.get())
            t = time.time() - t0
            on = {n: e for n, _, e in self.CHANNELS}
            angles = [0.2 * math.sin(2 * math.pi * 0.4 * t + ph) for ph in (0.0, 1.3, 2.1)]
            if random.random() < 0.004:
                airborne ^= 1
                hub.add_line("Propulsion ended; attitude control enabled" if airborne
                             else "Landing detected; attitude control disabled")
            if on["imu_age"]:
                hub.add_line(f">QuatAgeMs:{random.randint(0, 3)}")
                hub.add_line(f">GyroAgeMs:{random.randint(0, 3)}")
            if on["lqr_torque"]:
                for axis, a in zip("xyz", angles):
                    hub.add_line(f">LQR-tau-{axis}:{-1.5 * a + random.gauss(0, 0.02):.4f}")
            if on["lqr_state"]:
                for axis, a in zip("xyz", angles):
                    hub.add_line(f">qe-{axis}:{a / 2:.4f}")
                for axis, ph in zip("xyz", (0.0, 1.3, 2.1)):
                    hub.add_line(f">omega-{axis}:{0.5 * math.cos(2 * math.pi * 0.4 * t + ph):.4f}")
            if on["flight"]:
                hub.add_line(f">AccelerationMagnitude:{9.81 + random.gauss(0, 0.05) - (9.0 if airborne else 0):.2f}")
                hub.add_line(f">Airborne:{airborne}")
            if on["euler"]:
                for name, a in zip(("Roll", "Pitch", "Yaw"), angles):
                    hub.add_line(f">{name}:{math.degrees(a):.2f}")
            time.sleep(self.rate_ms / 1000)


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def make_handler(hub: Hub, control: ChannelControl):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        def _send(self, code, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json_body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if not 0 < n <= 4096:
                return {}
            try:
                data = json.loads(self.rfile.read(n))
                return data if isinstance(data, dict) else {}
            except ValueError:
                return {}

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, INDEX_HTML.read_bytes(), "text/html; charset=utf-8")
            elif path == "/events":
                self._events()
            elif path == "/export.csv":
                name = f"telemetry_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
                self._send(200, hub.export_csv().encode(), "text/csv",
                           {"Content-Disposition": f'attachment; filename="{name}"'})
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if path == "/clear":
                hub.clear()
                ok = True
            elif path == "/record":
                hub.set_recording(bool(self._json_body().get("on")))
                ok = True
            elif path == "/tlm":
                body = self._json_body()
                if "rate" in body:
                    ok = isinstance(body["rate"], (int, float)) and control.set_rate(body["rate"])
                else:
                    ok = isinstance(body.get("channel"), str) and control.set_channel(body["channel"], bool(body.get("on")))
            elif path == "/send":
                text = str(self._json_body().get("text", "")).strip()
                ok = bool(text) and len(text) < 200 and "\n" not in text and control.send(text)
                if ok:
                    hub.add_line(f"[sent] {text}")
            else:
                self._send(404, b"not found", "text/plain")
                return
            self._send(200 if ok else 409, b"ok" if ok else b"not sent (no device?)", "text/plain")

        def _events(self):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = hub.subscribe()
            try:
                self.wfile.write(f"data: {hub.snapshot()}\n\n".encode())
                self.wfile.flush()
                while True:
                    try:
                        msg = q.get(timeout=15)
                        self.wfile.write(f"data: {msg}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                hub.unsubscribe(q)
                self.close_connection = True

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port (default: auto-detect Teensy)")
    ap.add_argument("--baud", type=int, default=115200, help="ignored by Teensy USB serial, kept for other boards")
    ap.add_argument("--host", default="0.0.0.0", help="HTTP bind address (default: all interfaces)")
    ap.add_argument("--http-port", type=int, default=8080)
    ap.add_argument("--history", type=float, default=300.0, help="seconds of history sent to newly opened pages")
    ap.add_argument("--max-points", type=int, default=50000, help="max points kept per series on the server")
    ap.add_argument("--record-dir", default=str(HERE / "recordings"), help="where CSV recordings go")
    ap.add_argument("--record", action="store_true",
                    help="start recording immediately (otherwise use the Record button)")
    ap.add_argument("--state-file", default=str(HERE / "channels.json"),
                    help="where the dashboard's channel selection is remembered")
    ap.add_argument("--demo", action="store_true", help="generate fake data instead of reading serial")
    args = ap.parse_args()

    if serial is None and not args.demo:
        sys.exit("pyserial is not installed:  pip install pyserial   (or sudo apt install python3-serial)")

    hub = Hub(args.history, args.max_points, Recorder(Path(args.record_dir)))
    control = ChannelControl(hub, Path(args.state_file))
    source = DemoLink(hub, control) if args.demo else SerialLink(hub, control, args.port, args.baud)

    threading.Thread(target=hub.flush_loop, daemon=True).start()
    threading.Thread(target=source.run, daemon=True).start()

    server = ThreadingHTTPServer((args.host, args.http_port), make_handler(hub, control))
    server.daemon_threads = True
    print(f"Dashboard: http://{args.host if args.host != '0.0.0.0' else '<this-host>'}:{args.http_port}")
    if args.record:
        hub.set_recording(True)
        print(f"Recording to {hub.recorder.path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
