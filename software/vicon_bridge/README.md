# Vicon bridge (ground truth for the telemetry dashboard)

`vicon_to_dashboard.py` reads the hopper's pose from the Vicon system and streams it over UDP to the
telemetry dashboard (`software/telemetry_dashboard`) on the Pi. There it is shown in the cube pane's
**Trajectory** view next to the Teensy's trajectory estimate, and it is recorded with everything else.

```
Vicon --DataStream SDK--> vicon_to_dashboard.py --UDP 47269--> Pi: telemetry_server.py --> browser
```

## Setup (Vicon PC)

1. Install the Vicon DataStream SDK. Its Python package is in the SDK's `Python` folder:
   `pip install "<DataStream SDK>/Python"` (this provides `vicon_dssdk`).
   **Use 64-bit Python 3.12 (or similar), not 3.15:** under Python 3.15 the SDK crashes the process with
   "Windows fatal exception: access violation". With several versions installed, use
   `py -3.12 -m pip install ...` and `py -3.12 vicon_to_dashboard.py ...`.
2. In Vicon Tracker/Nexus, create an object (subject) for the hopper. **Set the object's local axes to match the
   IMU's body axes**, at least the x axis, so the dashboard can derive the heading offset from the attitude
   (see below), and so the orientation can be compared with the IMU later.
3. Run:

```bash
python vicon_to_dashboard.py --dashboard raspberrypi.local --subject hopper
```

| option | default | |
|---|---|---|
| `--dashboard HOST` | (required) | the Pi running `telemetry_server.py` |
| `--udp-port N` | 47269 | must match `telemetry_server.py --udp-port` |
| `--vicon HOST:PORT` | `localhost:801` | Vicon DataStream server |
| `--subject NAME` | first subject | Vicon object to stream |
| `--segment NAME` | root segment | |
| `--max-rate HZ` | every frame | limit the send rate |
| `--fake` | off | send a fake hop instead of reading Vicon (no SDK needed) |

To just check the Vicon stream without the dashboard, `vicon_print.py` prints the pose in the terminal
(position, quaternion, roll/pitch/yaw, latency); it takes `--vicon`, `--subject`, `--segment`, `--rate HZ`
(default 10 lines/s, 0 = every frame) and `--fake`.

The script reconnects automatically if the Vicon server restarts. Status messages (connected, subject picked,
occluded / tracked again) appear in the dashboard console as `[udp <ip>] Vicon: ...`.

## What is sent (one UDP datagram per Vicon frame)

| name | unit | |
|---|---|---|
| `ViconX`, `ViconY`, `ViconZ` | m | segment position in the Vicon world frame (Z up) |
| `ViconQw`, `ViconQx`, `ViconQy`, `ViconQz` | – | segment orientation, scalar first, segment -> Vicon world |
| `ViconOccluded` | 0/1 | 1: not tracked this frame (no pose sent) |
| `ViconLatencyMs` | ms | Vicon's processing latency for the frame |

All of these values are written to the dashboard's CSV recordings (Record button) and to Export CSV. The orientation is
not drawn. It is there to compare with the IMU attitude (`q-w..q-z`) offline.

## Alignment in the dashboard

The trajectory estimate is in ENU with the launch point as origin. The dashboard maps Vicon into that frame per hop:

    p_ENU = Rz(offset) · (p_vicon − p_vicon(launch))

* **Origin:** the Vicon position at the launch, which is the first sample of the burn.
* **Heading offset:** the **Vicon hdg** field in the Trajectory view (degrees, saved in the browser). **from attitude**
  sets it so the Vicon object's heading matches the IMU heading at the current (or paused) moment. Use it with the hopper
  standing level and the Vicon object's x axis defined along the IMU x axis. You can also type in a known value.

The Vicon path is drawn in purple from the launch until shortly after the predicted landing. After landing, the header
shows **Δland vs Vicon**, the horizontal distance between the predicted landing point and where Vicon saw the hopper
end up.

Timestamps are taken on the Pi when the data arrives, for the Teensy and Vicon alike. Their relative timing is good to
a few ms on a LAN, plus Vicon's own latency (`ViconLatencyMs`).

To test without hardware, run `telemetry_server.py --demo`. It includes fake Vicon data in a frame rotated by 30°, so
**from attitude** should give 30. Use `--demo-no-vicon` to test this script against the demo instead
(`--fake`).
