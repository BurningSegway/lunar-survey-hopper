#!/usr/bin/env python3
"""Send values to the telemetry dashboard over UDP (template for the Vicon script).

The dashboard server (telemetry_server.py) listens on UDP port 47269 for Teleplot
lines, the same ">name:value" format the Teensy prints. Send one datagram per
motion-capture frame so all values of that frame get the same timestamp on the
server:

    from udp_sender_example import DashboardSender
    dash = DashboardSender("raspberrypi.local")      # the Pi running telemetry_server.py
    dash.send(ViconX=x, ViconY=y, ViconZ=z)          # metres, once per frame
    dash.log("Vicon: subject 'hopper' tracked")      # optional console message

Only the Python standard library is needed. Run this file directly to stream a
fake hop at 100 Hz for testing:

    python3 udp_sender_example.py raspberrypi.local
"""

from __future__ import annotations

import argparse
import math
import socket
import time

DEFAULT_PORT = 47269


class DashboardSender:
    def __init__(self, host: str, port: int = DEFAULT_PORT):
        self.addr = (socket.gethostbyname(host), port)   # resolve once, not per frame
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, **values: float):
        """One datagram with one line per value. Names: letters, digits, '_' or '-'."""
        lines = [f">{name}:{float(v):.5f}" for name, v in values.items()
                 if v is not None and math.isfinite(float(v))]
        if lines:
            self.sock.sendto("\n".join(lines).encode(), self.addr)

    def log(self, text: str):
        """A console line in the dashboard, shown as "[udp <sender ip>] text"."""
        self.sock.sendto(text.encode(), self.addr)


def fake_hop(dash: DashboardSender, rate_hz: float):
    """Rest 3 s, then a drag-free arc (v0 = 4.5 m/s, 75 deg elevation), repeated."""
    g, v0, elev = 9.81, 4.5, math.radians(75)
    vh, vu = v0 * math.cos(elev), v0 * math.sin(elev)
    flight = 2 * vu / g
    dash.log("fake Vicon stream started")
    t0, n = time.time(), 0
    while True:
        tc = (time.time() - t0) % (3.0 + flight)
        tau = max(0.0, tc - 3.0)
        dash.send(ViconX=1.0 + vh * tau, ViconY=2.0, ViconZ=0.1 + vu * tau - 0.5 * g * tau * tau)
        n += 1
        time.sleep(max(0.0, t0 + n / rate_hz - time.time()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("host", nargs="?", default="127.0.0.1", help="machine running telemetry_server.py")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--rate", type=float, default=100.0, help="frames per second")
    args = ap.parse_args()
    fake_hop(DashboardSender(args.host, args.port), args.rate)
