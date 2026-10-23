# Python MuJoCo jumper simulator

The supported simulator is the Python application in `jumper_sim/`, using
MuJoCo's interactive Python viewer and the model at
`build/Jumping_Robot.xml`. Its controls run the jumper's PID or LQR attitude
controller, jump/tilt behavior, and reaction-wheel management while the viewer
is open.

## Setup

MuJoCo 3.2.3 has a prebuilt wheel for Python 3.12. Create the project venv and
install the runtime and terrain conversion dependencies:

```sh
python3.12 -m venv --without-pip .venv
pip --python .venv/bin/python install -r requirements.txt -r requirements-terrain.txt
```

The MJCF uses a visual `build/terrain.obj` and a collision
`build/terrain_height.png`, generated locally from the USD source. Place
`terrain_merged.usd` in `build/` and run:

```sh
.venv/bin/python build/convert_usd.py
```

The converter also accepts explicit paths:

```sh
.venv/bin/python build/convert_usd.py --input /path/to/terrain.usd --output build/terrain.obj --heightfield build/terrain_height.png
```

Terrain source and generated OBJ/PNG are not checked into Git. Generate both
terrain assets before launching the model. The converter requires the packages
listed in `requirements-terrain.txt`. The visual mesh remains at its original
orientation and `z=0` placement. Its collision is approximated by a heightfield;
there is no separate plane geom in the MJCF.

## Launch

Run from this directory:

```sh
.venv/bin/python -m jumper_sim --controller pid
```

Options:

- `--controller pid|lqr` selects the attitude controller (default: `pid`).
- `--model PATH` loads another compatible MJCF (default:
  `build/Jumping_Robot.xml`). It must provide the simulator's named root,
  reaction-wheel, jump, and tilt joints/actuators.

## Controls

- `Space`: initiate a jump (release for 0.4 seconds, then wind the spring back)
- `Enter`: toggle front-leg shortening force for directional tilt
- `Backspace`: toggle Earth/Moon gravity without resetting the simulation
- When the robot remains below the configured linear and angular speed limits
  (0.05 m/s and 0.05 rad/s by default) for one second, wheel speed references
  ramp toward zero while existing attitude references are retained. Motion
  resumes normal stabilization.
- `X`: disable reaction-wheel motor torque; press `C` to re-enable stabilization
- `C`: re-enable stabilization after disabling it with `X`
- `R`: reset the simulation and restore Earth gravity

The separate state window shows a red `RELAXATION INACTIVE` button while
normal stabilization is active, and turns green when automatic wheel
relaxation begins.

MuJoCo's viewer has its own Backspace reset shortcut. The simulator restores
the previous physics state when that default action occurs, then applies the
gravity toggle; use `R` when an actual reset is desired.

The simulator streams telemetry over UDP to PlotJuggler at
`127.0.0.1:9870`. Open `PlotJuggler_Setup.xml` in PlotJuggler to load the
existing orientation and reaction-wheel plots. The stream includes current
roll, pitch, and yaw, their references, and X/Y/Z wheel speeds in rad/s.
Start PlotJuggler's UDP Server listening on port `9870` before launching the
simulator.

## Contact gating (LQR only)

The LQR only controls attitude in flight. From the first step any part of the
robot touches the terrain it is "in contact": attitude control stops, the
integrator resets and the wheel-rest ramp bleeds the wheels' momentum into
the ground (this replaces the stationary-for-1-s rest trigger for the LQR).
It is airborne again after 60 ms without contact (`FLIGHT_DEBOUNCE`), so
rocking on the legs or skidding is one contact phase; at lift-off the LQR
resumes with a fresh integrator and the current heading as yaw reference.
The PID keeps its previous behaviour. Disable with
`JumperSimulation(..., contact_gating=False)`.

## Landing reference from the depth camera (LQR only)

A RealSense D435 stand-in (site `realsense` on the bottom face) looks down.
While the robot is airborne it produces 30 Hz depth frames (106x60 rays,
87x58 deg, 0.3-10 m, D435 stereo noise, 50 ms latency) by ray casting the
terrain heightfield. Each frame is fitted to the area-averaged ground plane
in a 0.5 m-radius disc straight below the robot and fused in a small Kalman
filter; fits whose depth noise is too large (above ~5-6 m) are ignored and
the reference freezes when touchdown is < 1 m or < 0.3 s away. The LQR holds
body z along that plane normal (clamped to 20 deg) with the usual yaw
reference. Nothing updates in contact; a jump resets the reference to level.
PlotJuggler's `roll_ref`/`pitch_ref` show it.

In the viewer: the disc is the latest fit (orange = it updated the
reference, grey = not used or held in contact), the green arrow is the
reference normal, and the blue dots are a sample of the depth points used.

- `jumper_sim/depth_camera.py`: camera spec, ray casters, simulated camera
  (the only simulation-specific part).
- `jumper_sim/landing_reference.py`: plane fit, filter and reference attitude;
  written to run unchanged on the robot from real D435 frames + IMU attitude.
- `jumper_sim/visualization.py`: the viewer overlay.
- Tuning: `LandingReferenceConfig` (disc radius, max tilt, filter noise,
  freeze rules) and `DepthCameraSpec`.

## Benchmarks

- `python -m benchmarks.lqr_benchmark --label NAME [--compare OTHER]`:
  free-flight attitude recovery; `python -m benchmarks.plot_results A B ...`
  plots saved runs.
- `python -m benchmarks.landing_benchmark --label NAME`: drops onto sloped
  terrain, touchdown misalignment with and without the camera reference.
- `python -m benchmarks.contact_benchmark --label NAME`: drops with bounces
  and a hop from the spawn point, with and without contact gating.

Contact behaviour differs between MuJoCo versions (on 3.2.3 the jump spring
launches the robot at ~5 m/s, on 3.16 at ~1 m/s), so run the contact and
landing benchmarks on the pinned version.

## Tuning

Edit `ReactionWheelTuning`, `PidGains`, and `LqrLimits` in
`jumper_sim/controller.py` to change the PID gains, wheel speed/acceleration
limits, stationary detection thresholds and duration, wheel-rest behavior,
wheel friction, and the LQR's physical tuning limits. The LQR takes the
robot's inertia from the MJCF; keep `robot_inertia`/`wheel_inertia` and
`wheel_viscous_friction` consistent with `build/Jumping_Robot.xml` for the
PID and the motor limits.


## Tests

Run controller, telemetry, and available headless MuJoCo tests with:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

The headless model tests require the locally generated
`build/terrain.obj` and `build/terrain_height.png` and are skipped with a setup
hint when either file is absent.
