from __future__ import annotations

import json
import socket
from typing import TypedDict

from .controller import ReactionWheelTuning, SensorData, WheelSpeeds


class TelemetryPayload(TypedDict):
    roll: float
    roll_ref: float
    pitch: float
    pitch_ref: float
    yaw: float
    yaw_ref: float
    vel_x: float
    vel_y: float
    vel_z: float
    sat_max: float
    sat_min: float


class UdpTelemetry:
    def __init__(self, host: str = "127.0.0.1", port: int = 9870) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._address = (host, port)

    def send(self, payload: TelemetryPayload) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._socket.sendto(encoded, self._address)

    def close(self) -> None:
        self._socket.close()


def make_telemetry_payload(
    sensor: SensorData,
    yaw_reference: float,
    wheel_speeds: WheelSpeeds,
    tuning: ReactionWheelTuning,
) -> TelemetryPayload:
    return {
        "roll": sensor.roll,
        "roll_ref": 0.0,
        "pitch": sensor.pitch,
        "pitch_ref": 0.0,
        "yaw": sensor.yaw,
        "yaw_ref": yaw_reference,
        "vel_x": wheel_speeds.x,
        "vel_y": wheel_speeds.y,
        "vel_z": wheel_speeds.z,
        "sat_max": tuning.max_wheel_velocity,
        "sat_min": -tuning.max_wheel_velocity,
    }
