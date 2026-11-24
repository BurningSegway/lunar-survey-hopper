#!/usr/bin/env python3
"""Stream a Vicon subject's pose to the hopper telemetry dashboard (ground truth).

Runs on the Vicon PC (or any machine that can reach the Vicon DataStream server),
reads one subject per frame with the Vicon DataStream SDK and sends it over UDP to
software/telemetry_dashboard/telemetry_server.py on the Pi, as Teleplot lines:

    ViconX ViconY ViconZ            position in the Vicon world frame [m] (Z up)
    ViconQw ViconQx ViconQy ViconQz orientation, scalar first, subject -> Vicon world
    ViconOccluded                   1 if the subject was not tracked in this frame
                                    (then no pose is sent for that frame)
    ViconLatencyMs                  Vicon's own processing latency for the frame

One datagram per Vicon frame, so the server stamps all values of a frame together.
The dashboard aligns the position to the trajectory estimate itself (launch point
as origin, heading offset set in the trajectory view); the orientation is only
recorded (Record button / Export CSV) for comparison with the IMU later.

Requirements: Python 3.8+ and the Vicon DataStream SDK Python package
(vicon_dssdk, installed from the SDK's "Python" folder: pip install <sdk>/Python).

    python vicon_to_dashboard.py --dashboard raspberrypi.local --subject hopper
    python vicon_to_dashboard.py --dashboard raspberrypi.local --fake      # no Vicon needed
"""

from __future__ import annotations

import argparse
import math
import socket
import sys
import time

DEFAULT_UDP_PORT = 47269   # telemetry_server.py --udp-port


class DashboardSender:
    def __init__(self, host: str, port: int = DEFAULT_UDP_PORT):
        print(f"Dashboard: resolving {host}", flush=True)
        try:
            ip = socket.gethostbyname(host)   # resolve once, not per frame
        except OSError as e:
            sys.exit(f"Dashboard: cannot resolve '{host}' ({e}). Use the Pi's IP address instead, "
                     f"e.g. --dashboard 192.168.1.42 (on the Pi: hostname -I)")
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"Dashboard: sending to {ip}:{port} (UDP)", flush=True)

        self.sent, self.errors, self.last_report = 0, 0, time.time()

    def send(self, **values: float):
        lines = [f">{name}:{float(v):.6f}" for name, v in values.items()
                 if v is not None and math.isfinite(float(v))]
        if lines:
            try:
                self.sock.sendto("\n".join(lines).encode(), self.addr)
                self.sent += 1
            except OSError as e:
                if self.errors == 0:
                    print(f"Dashboard: send to {self.addr[0]}:{self.addr[1]} failed ({e})", flush=True)
                self.errors += 1
        self.report()

    def report(self, every_s: float = 5.0):
        """Terminal status every few seconds, so a silent stream is visible."""
        now = time.time()
        if now - self.last_report >= every_s:
            dt = now - self.last_report
            print(f"Dashboard: {self.sent} frames sent to {self.addr[0]}:{self.addr[1]} "
                  f"in the last {dt:.0f} s ({self.sent / dt:.0f} Hz)"
                  + (f", {self.errors} send errors" if self.errors else ""), flush=True)
            self.sent, self.errors, self.last_report = 0, 0, now

    def log(self, text: str):
        """Console line in the dashboard ("[udp <this ip>] text")."""
        print(text, flush=True)
        try:
            self.sock.sendto(text.encode(), self.addr)
        except OSError as e:
            print(f"Dashboard: send failed ({e})", flush=True)


# --------------------------------------------------------------------------- #
# Vicon DataStream SDK
# --------------------------------------------------------------------------- #
def run_vicon(dash: DashboardSender, server: str, subject: str | None, segment: str | None, max_rate: float):
    print("Vicon: loading the DataStream SDK (vicon_dssdk)", flush=True)
    try:
        from vicon_dssdk import ViconDataStream
    except ImportError as e:
        sys.exit(f"vicon_dssdk could not be imported ({e}): install the Vicon DataStream SDK Python "
                 "package (pip install <DataStream SDK>/Python), or use --fake to test without Vicon")

    min_dt = 1.0 / max_rate if max_rate > 0 else 0.0
    while True:  # (re)connect loop
        client = ViconDataStream.Client()
        dash.log(f"Vicon: connecting to {server}")
        attempts = 0
        while True:
            try:
                client.Connect(server)
                if client.IsConnected():
                    break
            except ViconDataStream.DataStreamException as e:
                attempts += 1
                if attempts % 5 == 1:
                    print(f"Vicon: cannot connect to {server} ({e}), retrying...", flush=True)
            time.sleep(1.0)
        client.EnableSegmentData()
        try:  # lowest latency: the server pushes every frame
            client.SetStreamMode(ViconDataStream.Client.StreamMode.EServerPush)
        except Exception:
            pass
        dash.log(f"Vicon: connected to {server}")

        subj, seg, last_sent, was_occluded, no_subject_msg = None, None, 0.0, None, 0.0
        try:
            while client.IsConnected():
                try:
                    if client.GetFrame() is False:
                        continue
                except ViconDataStream.DataStreamException:
                    time.sleep(0.001)
                    continue

                if subj is None:  # pick the subject / segment once they appear
                    names = client.GetSubjectNames()
                    if not names:
                        if time.time() - no_subject_msg > 5.0:
                            dash.log("Vicon: no subjects in the stream yet (is the object enabled/tracked in Tracker?)")
                            no_subject_msg = time.time()
                        continue
                    if subject and subject not in names:
                        dash.log(f"Vicon: subject '{subject}' not found, available: {', '.join(names)}")
                        time.sleep(2.0)
                        continue
                    subj = subject or names[0]
                    seg = segment or client.GetSubjectRootSegmentName(subj)
                    dash.log(f"Vicon: streaming subject '{subj}', segment '{seg}'")

                now = time.time()
                if now - last_sent < min_dt:
                    continue
                last_sent = now

                (px, py, pz), occluded = client.GetSegmentGlobalTranslation(subj, seg)
                (qx, qy, qz, qw), q_occluded = client.GetSegmentGlobalRotationQuaternion(subj, seg)
                occluded = bool(occluded or q_occluded)
                try:
                    latency_ms = client.GetLatencyTotal() * 1000.0
                except Exception:
                    latency_ms = None
                if occluded != was_occluded:
                    if occluded:
                        dash.log(f"Vicon: '{subj}' occluded")
                    elif was_occluded is not None:
                        dash.log(f"Vicon: '{subj}' tracked again")
                    was_occluded = occluded
                if occluded:
                    dash.send(ViconOccluded=1, ViconLatencyMs=latency_ms)
                else:
                    dash.send(ViconX=px / 1000.0, ViconY=py / 1000.0, ViconZ=pz / 1000.0,  # mm -> m
                              ViconQw=qw, ViconQx=qx, ViconQy=qy, ViconQz=qz,
                              ViconOccluded=0, ViconLatencyMs=latency_ms)
        except ViconDataStream.DataStreamException as e:
            dash.log(f"Vicon: connection lost ({e}), reconnecting")
        print("Vicon: disconnected, reconnecting", flush=True)
        try:
            client.Disconnect()
        except Exception:
            pass
        time.sleep(1.0)


# --------------------------------------------------------------------------- #
# Fake data (no Vicon needed)
# --------------------------------------------------------------------------- #
def run_fake(dash: DashboardSender, rate_hz: float):
    """3 s rest, then a drag-free hop (v0 = 4.5 m/s, 75 deg elevation), repeated."""
    g, v0, elev = 9.81, 4.5, math.radians(75)
    vh, vu = v0 * math.cos(elev), v0 * math.sin(elev)
    flight = 2 * vu / g
    dash.log("Vicon: fake stream started")
    t0, n = time.time(), 0
    while True:
        tau = max(0.0, (time.time() - t0) % (3.0 + flight) - 3.0)
        dash.send(ViconX=1.0 + vh * tau, ViconY=2.0, ViconZ=0.1 + vu * tau - 0.5 * g * tau * tau,
                  ViconQw=1.0, ViconQx=0.0, ViconQy=0.0, ViconQz=0.0, ViconOccluded=0)
        n += 1
        time.sleep(max(0.0, t0 + n / rate_hz - time.time()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dashboard", required=True, help="host running telemetry_server.py, e.g. raspberrypi.local")
    ap.add_argument("--udp-port", type=int, default=DEFAULT_UDP_PORT, help="telemetry_server.py --udp-port")
    ap.add_argument("--vicon", default="localhost:801", help="Vicon DataStream server host:port")
    ap.add_argument("--subject", help="Vicon subject (object) name; default: the first one streamed")
    ap.add_argument("--segment", help="segment name; default: the subject's root segment")
    ap.add_argument("--max-rate", type=float, default=0,
                    help="limit the frames sent per second (default 0 = every Vicon frame)")
    ap.add_argument("--fake", action="store_true", help="send a fake hop instead of reading Vicon")
    args = ap.parse_args()

    print(f"vicon_to_dashboard: Python {sys.version.split()[0]} ({sys.executable})", flush=True)
    dash = DashboardSender(args.dashboard, args.udp_port)
    try:
        if args.fake:
            run_fake(dash, args.max_rate or 100.0)
        else:
            run_vicon(dash, args.vicon, args.subject, args.segment, args.max_rate)
    except KeyboardInterrupt:
        dash.log("Vicon: stream stopped")


def _own_console_window() -> bool:
    """True when this process got its own console window (e.g. double-clicked on
    Windows), which closes as soon as the script exits and hides any error."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        return ctypes.windll.kernel32.GetConsoleProcessList((ctypes.c_uint * 2)(), 2) <= 1
    except Exception:
        return False


if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()   # print where it died if the Vicon SDK's DLL crashes the process
    try:
        main()
    except SystemExit as e:
        if e.code not in (None, 0):
            if isinstance(e.code, str):
                print(e.code, file=sys.stderr)
            else:
                print(f"exited with code {e.code}", file=sys.stderr)
        if _own_console_window():
            input("\nPress Enter to close...")
        raise SystemExit(0 if e.code in (None, 0) else 1)
    except BaseException:
        import traceback
        traceback.print_exc()
        if _own_console_window():
            input("\nPress Enter to close...")
        raise SystemExit(1)
