#!/usr/bin/env python3
"""Print a Vicon subject's pose in the terminal (no dashboard needed).

Same Vicon reading as vicon_to_dashboard.py, but every frame is printed instead of
sent over UDP. Useful to check that the Vicon stream, subject name and axes are
right before a test. Each line shows:

    frame   Vicon frame number
    x y z   position in the Vicon world frame [m] (Z up)
    qw qx qy qz   orientation, scalar first, subject -> Vicon world
    roll pitch yaw   the same orientation as ZYX Euler angles [deg]
    lat     Vicon's own processing latency for the frame [ms]

Occluded frames are printed as "OCCLUDED".

Requirements: Python 3.8+ and the Vicon DataStream SDK Python package
(vicon_dssdk, installed from the SDK's "Python" folder: pip install <sdk>/Python).

    python vicon_print.py --subject hopper
    python vicon_print.py --vicon 192.168.1.10:801 --rate 50
    python vicon_print.py --fake      # no Vicon needed
"""

from __future__ import annotations

import argparse
import math
import sys
import time


def quat_to_euler_deg(qw: float, qx: float, qy: float, qz: float) -> tuple[float, float, float]:
    """ZYX (yaw-pitch-roll) Euler angles in degrees from a unit quaternion."""
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def print_pose(frame, x, y, z, qw, qx, qy, qz, latency_ms=None):
    roll, pitch, yaw = quat_to_euler_deg(qw, qx, qy, qz)
    lat = f"{latency_ms:6.1f}" if latency_ms is not None else "     -"
    print(f"{frame:>8}  x {x:+8.4f}  y {y:+8.4f}  z {z:+8.4f} m  "
          f"q [{qw:+.4f} {qx:+.4f} {qy:+.4f} {qz:+.4f}]  "
          f"rpy {roll:+7.2f} {pitch:+7.2f} {yaw:+7.2f} deg  lat {lat} ms", flush=True)


def print_occluded(frame, subject, latency_ms=None):
    lat = f"{latency_ms:6.1f}" if latency_ms is not None else "     -"
    print(f"{frame:>8}  OCCLUDED ('{subject}' not tracked)  lat {lat} ms", flush=True)


# --------------------------------------------------------------------------- #
# Vicon DataStream SDK
# --------------------------------------------------------------------------- #
def run_vicon(server: str, subject: str | None, segment: str | None, rate: float):
    print("Vicon: loading the DataStream SDK (vicon_dssdk)", flush=True)
    try:
        from vicon_dssdk import ViconDataStream
    except ImportError as e:
        sys.exit(f"vicon_dssdk could not be imported ({e}): install the Vicon DataStream SDK Python "
                 "package (pip install <DataStream SDK>/Python), or use --fake to test without Vicon")
    print("Vicon: SDK loaded", flush=True)

    min_dt = 1.0 / rate if rate > 0 else 0.0
    while True:  # (re)connect loop
        print("Vicon: creating client", flush=True)
        client = ViconDataStream.Client()
        print(f"Vicon: connecting to {server}", flush=True)
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
        print(f"Vicon: connected to {server}", flush=True)

        subj, seg, last_print = None, None, 0.0
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
                        continue
                    if subject and subject not in names:
                        print(f"Vicon: subject '{subject}' not found, available: {', '.join(names)}", flush=True)
                        time.sleep(2.0)
                        continue
                    subj = subject or names[0]
                    seg = segment or client.GetSubjectRootSegmentName(subj)
                    print(f"Vicon: printing subject '{subj}', segment '{seg}' "
                          f"(available subjects: {', '.join(names)})", flush=True)

                now = time.time()
                if now - last_print < min_dt:
                    continue
                last_print = now

                try:
                    frame = client.GetFrameNumber()
                except Exception:
                    frame = "-"
                try:
                    latency_ms = client.GetLatencyTotal() * 1000.0
                except Exception:
                    latency_ms = None

                (px, py, pz), occluded = client.GetSegmentGlobalTranslation(subj, seg)
                (qx, qy, qz, qw), q_occluded = client.GetSegmentGlobalRotationQuaternion(subj, seg)
                if occluded or q_occluded:
                    print_occluded(frame, subj, latency_ms)
                else:
                    print_pose(frame, px / 1000.0, py / 1000.0, pz / 1000.0,  # mm -> m
                               qw, qx, qy, qz, latency_ms)
        except ViconDataStream.DataStreamException as e:
            print(f"Vicon: connection lost ({e}), reconnecting", flush=True)
        try:
            client.Disconnect()
        except Exception:
            pass
        time.sleep(1.0)


# --------------------------------------------------------------------------- #
# Fake data (no Vicon needed)
# --------------------------------------------------------------------------- #
def run_fake(rate_hz: float):
    """3 s rest, then a drag-free hop (v0 = 4.5 m/s, 75 deg elevation), repeated."""
    g, v0, elev = 9.81, 4.5, math.radians(75)
    vh, vu = v0 * math.cos(elev), v0 * math.sin(elev)
    flight = 2 * vu / g
    print("Vicon: fake stream started", flush=True)
    t0, n = time.time(), 0
    while True:
        tau = max(0.0, (time.time() - t0) % (3.0 + flight) - 3.0)
        print_pose(n, 1.0 + vh * tau, 2.0, 0.1 + vu * tau - 0.5 * g * tau * tau,
                   1.0, 0.0, 0.0, 0.0)
        n += 1
        time.sleep(max(0.0, t0 + n / rate_hz - time.time()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vicon", default="localhost:801", help="Vicon DataStream server host:port")
    ap.add_argument("--subject", help="Vicon subject (object) name; default: the first one streamed")
    ap.add_argument("--segment", help="segment name; default: the subject's root segment")
    ap.add_argument("--rate", type=float, default=10.0,
                    help="max lines printed per second (default 10; 0 = every Vicon frame)")
    ap.add_argument("--fake", action="store_true", help="print a fake hop instead of reading Vicon")
    args = ap.parse_args()

    print(f"vicon_print: Python {sys.version.split()[0]} ({sys.executable})", flush=True)
    try:
        if args.fake:
            run_fake(args.rate or 100.0)
        else:
            run_vicon(args.vicon, args.subject, args.segment, args.rate)
    except KeyboardInterrupt:
        print("\nVicon: stopped")


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
            print(e.code if isinstance(e.code, str) else f"exited with code {e.code}", file=sys.stderr)
        if _own_console_window():
            input("\nPress Enter to close...")
        raise SystemExit(0 if e.code in (None, 0) else 1)
    except BaseException:
        import traceback
        traceback.print_exc()
        if _own_console_window():
            input("\nPress Enter to close...")
        raise SystemExit(1)
