# Telemetry dashboard (Teleplot-style, in the browser)

This tool runs on the Raspberry Pi that is connected to the Teensy over USB. It reads the
Teensy's serial output and serves a live plotting dashboard that any device on the
network can open in a browser.

```
Teensy --USB serial--> Pi: telemetry_server.py --HTTP/SSE--> browser(s)
```

* Uses the **Teleplot format** the firmware already prints: `>name:value`
  (also `>name:ts:value`, `>name:v§unit|flags`, `>name:1;2;3`).
* All other lines (`ODrives running!`, warnings, ...) appear in the **console** at the bottom.
* **Auto-detects the Teensy** (USB VID 0x16C0, falling back to `/dev/ttyACM*`) and
  **reconnects automatically** after a reset, a re-flash or an unplug.
* **Recording is off by default.** Press **Record** in the page (or start with `--record`) to write
  `recordings/telemetry_<date>.csv` on the Pi, one `timestamp_s,name,value` row per sample. Each time you start
  recording, a new file is created. The file starts with the Teensy's channel list, so you can see which prints were on.
* The page needs **no internet access**: it has no CDN or external libraries, so it works when the
  Pi runs its own Wi-Fi hotspot in the field.
* Several browsers can watch at once. A newly opened page receives the last 5 minutes of history.

## Setup on the Pi

```bash
sudo apt install python3-serial      # or: pip install -r requirements.txt
sudo usermod -aG dialout $USER       # serial port access; log out and back in afterwards
python3 telemetry_server.py
```

Then open `http://<pi-hostname-or-ip>:8080`, for example `http://raspberrypi.local:8080`.

Note: only one program can own the serial port. Close the PlatformIO serial monitor or
Teleplot if they are open on the same port. To flash the Teensy from the Pi, stop the
dashboard first.

### Options

| flag | default | |
|---|---|---|
| `--port /dev/ttyACM0` | auto | serial device |
| `--http-port 8080` | 8080 | web port |
| `--history 300` | 300 s | history sent to newly opened pages |
| `--record` | off | start recording right away (otherwise use the Record button) |
| `--record-dir DIR` | `./recordings` | where CSV recordings are written |
| `--demo` | off | fake LQR-like data, for trying the UI without hardware |

### Start on boot

Edit the `User` and paths in `telemetry-dashboard.service`, then follow the instructions
in the comments at the top of that file.

## Using the dashboard

* **Click** a variable in the left list to open a new chart for it.
* **Drag** a variable onto an existing chart to overlay it.
* Click a **legend entry** to remove that series from its chart. Use × to close a chart.
* **Hover** over a chart to read the values at that point in time.
* **Space** (or the Pause button) freezes the view while data keeps being collected.
* **Window** sets the visible time span. **Columns** sets the chart grid layout.
* The chart layout is saved in the browser (localStorage), so it comes back after a reload.
* **Cube** opens a 3D attitude view. It shows the hopper as a cube rotated by the attitude the Teensy reports, inside the
  IMU's inertial frame (Xsens default ENU: X = East, Y = North, Z = Up, dashed axes over a ground grid). Body axes
  are solid (x red, y green, z blue), and the faces are labelled ±X/±Y/±Z.
  It uses the `quat` channel (q-w..q-z) when that is printing, and falls back to `euler` (Roll/Pitch/Yaw, Z-Y-X).
  If neither is on, opening the cube switches `quat` on. Drag to orbit the view, and double-click to reset it.
  When paused, the cube shows the paused moment. Hovering over any chart shows the attitude at that point in time.
* **Record / Stop recording** writes everything that arrives (all variables plus console lines) to a file on the Pi.
* **Export CSV** downloads the server's in-memory history as a wide table, like Teleplot's export. This covers all
  variables since the server started or since the last **Clear**, up to 50,000 points per variable
  (`--max-points`). It does **not** depend on the chart window or on which variables are plotted.
  **Clear** wipes that history for all viewers. Clear does not affect a running recording.

### Choosing what the Teensy prints

The LQR firmware groups its prints into **channels**, defined in
`teensy-firmware/teensy-lqr-stabilizer/include/telemetry.h` and in the table at the top of `main.cpp`.
The **Teensy prints** panel at the top of the sidebar lists them:

* Use a checkbox to turn a channel on or off. The Teensy confirms each change, and the panel then shows its real state.
* The rate selector sets the Teensy's print rate for all channels (default 50 Hz).
* The variable list also shows variables whose channel is **off** (greyed out). Clicking one
  switches its channel on and opens a chart for it.
* Your selection is saved in `channels.json`. It is **re-applied automatically** whenever the Teensy
  reboots or reconnects, so you don't need to re-flash to change what is printed.

The console's input box sends any line to the Teensy. The Teensy understands these commands, which also work from any serial monitor:

```
tlm?              list channels (reply: "#tlm {json}")
tlm <name> 1|0    channel on/off, e.g. "tlm lqr_state 1"
tlm all 1|0       every channel on/off
tlm rate <ms>     print period, 10..1000 ms
```

To add a new print in the firmware, add an entry to `TelemetryChannelId`, add a matching row to
`telemetryChannels[]`, and add the `Telemetry::value(...)` calls in the telemetry block at the end of `loop()`.
The dashboard picks the new channel up automatically.

Timestamps are taken on the Pi when each line arrives. A device timestamp in `>name:ts:value`
is ignored.

## Performance note

The firmware prints at 50 Hz, about 8 values per tick (around 400 lines/s). The Pi handles this easily. The
server sends updates to browsers in batches about 20 times a second, and dense plots are decimated
to one min/max pair per pixel column.
