# Sim_MuJoCo: notes for Claude

MuJoCo simulation of a small lunar survey hopper. It jumps with a wound-up
spring and controls its attitude in flight with three reaction wheels. The aim
is to design and test mid-flight attitude control (and, later, port it to the
real robot). `README.md` is the user-facing reference and is kept up to date;
read it for options, controls and measured results.

## Commands

```sh
.venv/bin/python -m jumper_sim --controller lqr          # viewer (default controller is pid)
.venv/bin/python -m unittest discover -s tests           # 82 tests, ~30 s, all must pass
.venv/bin/python -m benchmarks.lqr_benchmark --label NAME      # free flight
.venv/bin/python -m benchmarks.landing_benchmark --label NAME  # drops onto slopes
.venv/bin/python -m benchmarks.contact_benchmark --label NAME  # bounces + a hop
pip --python .venv/bin/python install -r requirements.txt  # venv has no pip of its own
```

Headless rendering, if ever needed: `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`.
Model tests need the generated `build/terrain.obj` and `build/terrain_height.bin`
(see README "Setup"); they are skipped without them.

## Hard constraints

- The simulation starts in Moon gravity (`JumperSimulation(gravity=...)`);
  tests that time an Earth hop pass `gravity=EARTH_GRAVITY`.
- **MuJoCo is pinned to 3.2.3** (`requirements.txt`). An upgrade to 3.16 was
  tried and fully rolled back: contact behaviour differs a lot between
  versions (jump height, sinking, sticking). Don't upgrade without being asked.
- Python 3.12 venv at `.venv`. `open3d` is only used by the point-cloud window.
- New flight features (contact gating, camera, hop reference, scanning) are
  **LQR only**. The PID path is kept as it was; its gains silently rely on the
  wheel bearing friction for damping, so don't add friction compensation to it.
- Keep the gyroscopic feedforward, the landing-filter outlier gate and all
  three benchmarks (the user chose to keep them).

## Layout (`jumper_sim/`)

| File | Role |
|---|---|
| `controller.py` | PID and per-axis LQI attitude controllers (CARE solved numerically, tuning in physical units via `LqrLimits`, Bryson's rule), bearing-friction and gyroscopic feedforward, wheel limits, wheel-rest ramp |
| `simulation.py` | `JumperSimulation`: one control step + `mj_step`; contact gating (attitude control only in flight, wheel momentum bleed in contact, 60 ms flight debounce); jump sequence; wires camera, flight phase, hop reference and scanning together; `CameraLandingReference` |
| `flight_phase.py` | Hop phase state machine GROUND / TAKEOFF / ASCENT / DESCENT / LANDING (jump command, 0.5 m above stance from integrated IMU vertical velocity, apex, landing-filter freeze or contact, still 0.5 s); all behaviours key off it. Starts in DESCENT |
| `flight_reference.py` | Hop reference: on a jump's ascent the downward camera (body -z) looks at a settable world target (tilt cap 60 deg); on descent back to level, then the landing plane. Slew limit 180 deg/s |
| `depth_camera.py` | Simulated RealSense D435 (106x60 rays, 87x58 deg, 0.3-10 m, stereo noise, 30 Hz, 50 ms latency) and an exact fast heightfield ray caster (matches `mj_multiRay`) |
| `landing_reference.py` | Area-weighted ground-plane fit under the robot, calibrated uncertainty (noise vs. roughness), Kalman filter on tilt with ignore/freeze rules near and far from the ground |
| `jump_aim.py` | On the ground, lean the body axis towards the track target (3 deg near, 8 deg at >= 30 m), closed loop on IMU tilt via a leg-shortening plane (integrator, only while still, held during release, reset at touchdown, legs capped at 10 cm, off on ground > 20 deg); motors `tilt_fl/fr/bl/br` |
| (target, tipped) | `simulation.py`: bunny `target_object` placed on the terrain at `--track-target` (offset from `mesh_pos`; parked 100 m down otherwise), seen by `MeshRayCaster` in the shared heightfield caster's `objects`; tipped (> 60 deg in GROUND) pulls all legs in. Righting is NOT possible with this hardware (tested: wheel push, wheel spin-brake, leg kick) |
| `pointcloud.py` | Target scan: on-target ascent frames to world points, binary PLY at `pointclouds/target_scan.ply`, emptied at program start, rewritten after each ascent |
| `cloud_viewer.py` | Live Open3D window watching that PLY (own process, started by `app.py`) |
| `visualization.py` | Overlays in the MuJoCo viewer: landing plane, target, scan sample |
| `app.py` | CLI and viewer loop |
| `telemetry.py`, `status_indicator.py` | PlotJuggler UDP stream (127.0.0.1:9870; layout in `PlotJuggler_Setup.xml`), rest indicator |

Model: `build/Jumping_Robot.xml`. Camera site `realsense` under the base;
terrain = moon crater field from `build/Testing/Temporary/moon-surface`
(100 x 100 m, cm -> m, Y-up -> Z-up, plain colour, no texture) made by
`build/convert_usd.py`, centred with the spawn ground at z=0; heightfield in geom group 2 with firm contact
(`solref 0.005 1.5`, `solimp 0.9 0.95 0.01 0.5 2`, friction 0.7; the old
sand-like `solref 0.03 2` let legs sink 8-17 cm on landing and get stuck); the jumping
foot is a firm sphere (`priority 2`, `solref 0.002 1`) overriding the sand;
jump spring 15000 N/m, wind-up force 13 kN (~3 m Earth / ~16 m Moon mean
apex, +-25 % with the spot).
`RW_PID.cpp` is the firmware-side PID, for the eventual port.

## Gotchas learned the hard way

- Right after `mj_step`, `qpos` is one step ahead of `xpos`/`site_xpos`. Tag
  sensor data with the pose it was computed from (`data.xpos`/`data.xquat` of
  the base body); mixing them put scan points ~2 cm off.
- MuJoCo's `accelerometer` sensor is ~1 m/s off when integrated, because of
  implicit joint damping. The sim uses the finite difference of body velocity
  as the IMU accelerometer.
- In free flight an accelerometer reads ~0 going up and coming down; the apex
  comes from the integrated vertical velocity, not from acceleration.
- In the soft sand the jump worked through a contact artifact (the foot sank
  in and was ejected): apex 0.01-49 m depending on the spot. Hence the firm
  foot. Ask before changing contact parameters.
- A flat cylinder on a flat heightfield touches at one rim point: off-centre
  push, up to 18 rad/s spin at take-off. Keep the foot round.
- Take-off spin above ~4 rad/s saturates the wheels (0.45 N*m*s each) and
  the ascent target tracking is lost.
- `mj_multiRay` has a different signature across versions (tests handle both).

## Working style the user expects

- Make one change at a time and show its measured effect (benchmarks or a
  short headless run) rather than just asserting an improvement.
- Run the full test suite after changes; add tests for new behaviour.
- Keep `README.md` in sync with features and CLI options.
- The user tests behaviour in the viewer themselves; say what to look for.

## Open items

- Phases still use simulator ground contact (impact, lift-off, attitude
  gating). On the robot: IMU impact/free fall (or leg compression sensors,
  if the hardware has them); LANDING has no behaviour of its own yet.

- Take-off spin from the foot slamming into the ground at release (~21 kN
  peak, foot held ~10 cm up while wound); with the firm ground up to 3 rad/s
  at some spots, and the two Earth hop tests fail on it.
- Robot tips over after touchdown on steep spots (contact benchmark), incl.
  the spawn hop's landing; attitude control is gated off in contact.
- Point-cloud processing (range crop, voxel downsample, registering jumps).
- Landing-spot prediction (could become the default track target).
- Sensor realism: IMU noise/bias/latency, attitude estimator, control rate.
- Hardware prep: measure bearing friction and inertia; port LQI + gating,
  landing reference and hop reference to the robot.
