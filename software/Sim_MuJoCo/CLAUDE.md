# Sim_MuJoCo: notes for Claude

MuJoCo simulation of a small lunar survey hopper. It jumps with a wound-up
spring and controls its attitude in flight with three reaction wheels. The aim
is to design and test mid-flight attitude control (and, later, port it to the
real robot). `README.md` is the user-facing reference and is kept up to date;
read it for options, controls and measured results.

## Commands

```sh
.venv/bin/python -m jumper_sim --controller lqr          # viewer (default controller is pid)
.venv/bin/python -m unittest discover -s tests           # ~146 tests, ~45 s, all must pass
.venv/bin/python -m benchmarks.lqr_benchmark --label NAME      # free flight
.venv/bin/python -m benchmarks.landing_benchmark --label NAME  # hops thrown onto slopes
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
- Both controllers run the same hop process (contact gating, lidar, hop
  reference, scanning); only the control law differs. The PID's gains
  silently rely on the wheel bearing friction for damping, so don't add
  friction compensation to it.
- Keep the landing-filter outlier gate and all three benchmarks (the user
  chose to keep them). The LQR is the project report's (ch. 4-5, user's
  choice, replacing the per-axis LQI with its gyroscopic feedforward) with
  the integral reference tracking of the user's MATLAB script
  `angular_reference_integral.mlx` (9-state LQI, eq. 25; not in the repo):
  keep it the report's model and the script's law and Bryson bounds (15 deg,
  60 deg/s, 5 deg*s, 10 rev/s^2 wheel acceleration as 0.26 N*m of torque;
  a test checks the gains against the script's). The script's N r
  feedforward is the error quaternion (N_x = [I; 0], N_u = 0). Anti-windup
  (5 deg integral zone, 0.8 N*m clamp, reset at lift-off/touchdown) is the
  user's choice after the pure integrator wound up. Linearised at
  Omega_hat = 0 (the report's choice, to save overhead), one fixed gain;
  `schedule_wheel_speeds=True` re-solves it at the measured wheel speeds.
- Design values (CAD J_c, J_w,i, 3.3 kg cube placeholder, four fjedre.dk
  13760 jump springs) live in `cube_model.py`; the MJCF is fitted to them by
  `build/fit_robot_model.py` (rerun it, don't hand-edit those MJCF values).

## Layout (`jumper_sim/`)

| File | Role |
|---|---|
| `cube_model.py` | Report ch. 4: `SENSOR_CUBE` (CAD J_c, J_w,i, mass), `JUMP_SPRING` x4, `ROBOT_MASS`; gyrostat `dynamics` f(x, u) (scalar-last quaternion), `jacobians` (4.18-4.19), `linearize` (reduced A_r, B_r at Omega_hat), `controllability_matrix`, `wheel_accelerations`. Input u = wheel motor torque tau: (J_c - J_w) w_dot = -tau - S(w) H (user's choice; replaced Omega_dot) |
| `controller.py` | PID and the report's LQR (ch. 5) as the script's LQI: u = tau = -K_bar [q_err_vec, w - w_ref, x_I] (torque straight to the motors, clipped at the motor limit), `augment`, `tracking_feedforward`, CARE by matrix sign function, Bryson's rule + anti-windup via `LqrLimits`, fixed gain at Omega_hat = 0 (optional rescheduling at measured wheel speeds); `reference_rate`, bearing-friction feedforward, wheel limits, wheel-rest ramp, `robot_inertia` (at qpos0 = the flight pose) |
| `simulation.py` | `JumperSimulation`: one control step + `mj_step`; contact gating (attitude control only in flight, wheel momentum bleed in contact, 60 ms flight debounce); jump sequence; wires lidar, flight phase, hop reference and scanning together; `LidarLandingReference` (lidar batches -> ground map, fits every 50 ms) |
| `flight_phase.py` | Hop phase state machine GROUND / TAKEOFF / ASCENT / DESCENT / LANDING (jump command, 0.5 m above stance from integrated IMU vertical velocity, apex, touchdown near (landing filter's near-touchdown rule) or contact, still 0.5 s); all behaviours key off it. Starts in DESCENT |
| `flight_reference.py` | Hop reference: on a jump's ascent body -z points at a settable world target, tilt cap 60 deg, and on into the descent for half the ascent's time (user's stand-in for a landing-spot criterion); then back to level, then the landing plane, updated until impact. Heading in ASCENT/DESCENT: `face_travel` (body +x along the horizontal velocity, <= 90 deg/s) or `spin` (`--spin`, any lidar; 45 deg/s, stopped 1 s before predicted touchdown); body -z unaffected. Slew limit 180 deg/s |
| `ray_casting.py` | Exact fast heightfield ray caster (matches `mj_multiRay`; skips 8x8-cell blocks the ray passes above their highest vertex, block looked up at the skipped segment's middle: ~5x fewer march steps, 2.4-3x faster; its cost is set by the number of march steps, i.e. ray length, more than by ray count) and a mesh ray caster for objects on it (einsum, not `@`: OpenBLAS threading made it 2-3x slower); site/geom names. The RealSense D435 depth camera that used to live here was removed (lidar only) |
| `landing_reference.py` | `GroundMap` of the jump's lidar points (0.5 m grid, emptied at each jump), ballistic landing-spot prediction (base within 0.3 m of the mapped ground), area-weighted plane fit of <= 1000 points within 0.5 m of the spot, calibrated uncertainty (noise vs. roughness), Kalman filter on tilt with ignore rules; keeps updating until impact (user's choice; `freeze_near_touchdown` restores the old freeze), its near-touchdown rule starts LANDING. Benchmarks/tests throw the robot onto a spot (starting in ASCENT with the IMU vertical velocity set) |
| `jump_aim.py` | On the ground, lean the body axis towards the track target (3 deg near, 8 deg at >= 30 m), closed loop on IMU tilt via a leg-shortening plane (integrator, only while still, held during release, reset at touchdown, legs capped at 10 cm, off on ground > 20 deg); motors `tilt_fl/fr/bl/br` |
| (target, tipped) | `simulation.py`: bunny `target_object` placed on the terrain at `--track-target` (offset from `mesh_pos`; parked 100 m down otherwise), seen by `MeshRayCaster` in the terrain caster's `objects` (`JumperSimulation._terrain`); tipped (> 60 deg in GROUND) pulls all legs in. Righting on the ground is NOT possible (tested: wheel push, wheel spin-brake, wheel rock, leg kick incl. 1000 N push) |
| `recovery.py` | Space while tipped: wheels spin up 0.15 s pushing the body into the ground, spring released, level reference in the air. ~75 % upright per press in Moon gravity, ~95 % within 3; 0 % on Earth. Outcome is very sensitive to timing (10 ms flips cases): judge changes on >= 50 poses (16 spots x 4 sides), not a handful |
| `pointcloud.py` | Lidar point cloud: every point of each tracked jump, lift-off to touchdown (per-point `range`), binary PLY at `pointclouds/target_scan.ply`, emptied at program start, rewritten at each touchdown (~21 MB per full-rate hop); touchdowns in `<name>.touchdowns.json` |
| `lidar.py` | The only sensor. Default and the one to develop on (user's choice): Livox Mid-360S on site `lidar`: +x side face, flush with the bottom, axis out of the face tilted 22.5 deg down (MJCF `euler="0 -67.5 0"`), so body -z is mid-band (band 38-97 deg off axis; blind cone around the axis); 2 cm range noise, 4 cm within 1 m; no self-occlusion. A bottom mount facing down (blind below, tracked 45 deg off the target) was tried and replaced: vertical drops tipped; Livox's recorded pattern from `mujoco-lidar` (R2 fill without it; it sweeps its elevation band in ~50 ms, so test over >= 50 ms), cast in 10 ms batches while airborne at `point_fraction` (1.0 with a GPU, 0.1 without). The user chose it over the Avia and the RealSense, which were removed. The other sensor under consideration, `--lidar as-dt1`: Sony AS-DT1 (a fifth of the Mid-360S's weight), 24x24 grid 30x30 deg (VFoV assumed), 15 fps whole frames, 20 m (datasheet outdoor at 50 %; less on regolith), 5 mm + 25 mm rounding, site `lidar_down` under the bottom tilted 10 deg towards +x; in flight the robot faces body +x along its travel (`LidarSpec.face_travel`). `--spin` (either lidar) spins slowly instead. Benchmarks take `--lidar`/`--spin` too |
| `gpu_ray_caster.py` | `WarpRayCaster`: the heightfield caster's triangles + objects on the GPU (NVIDIA Warp BVH), identical hits, ~80x faster (`gpu_available()`; numpy fallback). Warp's first init runs `uname -p` via Popen (tests that mock Popen must filter) |
| `cloud_viewer.py` | Live Open3D window watching that PLY and `<name>.touchdowns.json` (green labelled spheres; points within 0.3 m of the target white and larger) (own process, started by `app.py`); GUI `SceneWidget`, not the classic Visualizer (one global point size) nor `O3DVisualizer` (no key callbacks in Python) |
| `visualization.py` | Overlays in the MuJoCo viewer: landing plane at the predicted spot, target, scan sample, `h = ...` label (base height above the last ground contact; MuJoCo 3.2.3 has no `set_texts`, so it is an `mjGEOM_LABEL`) |
| `app.py` | CLI and viewer loop (`--smooth`: Mid-360S at 3 % of its point rate, measured to run without stutter on the CPU): sim paced to the wall clock, viewer synced at 120 Hz; `RobotFollower` centres a free camera on the robot each sync (MuJoCo's tracking camera moves 20 % per *drawn* frame from the viewer's own thread: it lagged and made a fast robot stutter) |
| `telemetry.py`, `status_indicator.py` | PlotJuggler UDP stream (127.0.0.1:9870; layout in `PlotJuggler_Setup.xml`), rest indicator |

Model: `build/Jumping_Robot.xml`. Lidar site `lidar` under the base;
terrain = moon crater field from `build/Testing/Temporary/moon-surface`
(100 x 100 m, cm -> m, Y-up -> Z-up, plain colour, no texture) made by
`build/convert_usd.py`, centred with the spawn ground at z=0; heightfield in geom group 2 with firm contact
(`solref 0.005 1.5`, `solimp 0.9 0.95 0.01 0.5 2`, friction 0.7; the old
sand-like `solref 0.03 2` let legs sink 8-17 cm on landing and get stuck); the jumping
foot is a firm sphere (`priority 2`, `solref 0.002 1`) overriding the sand;
jump: four springs (6680 N/m together), wound to full compression (257 mm,
221 J) at q = 0 with the foot at the bottom of the cube (user's choice:
spring length/placement not modelled for now), released to free length, wound
back after 0.4 s; damping 2, stiff stroke limits, a third of the spring mass
as armature; wind-up 1.25x the wound force (~1.65 m Earth / ~9.2 m Moon apex;
the foot falls 12.75 cm before touching ground, which costs most of the
energy). Robot 4.196 kg, J = diag(J_c) in flight (foot wound), spawn 1 cm
above the ground (starting in it, the firm foot ejects the robot).
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
- Test windows (Open3D, MuJoCo) open on the user's desktop: their mouse can
  move the view while you screenshot it, and offscreen Open3D rendering is
  black on this machine. Prefer headless checks; tell the user before
  opening windows.

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

- Robot tips over after touchdown on steep spots (contact benchmark), incl.
  the spawn hop's landing; attitude control is gated off in contact.
- Point-cloud processing (range crop, voxel downsample, registering jumps).
- Landing-spot prediction (could become the default track target). The
  landing reference already predicts it ballistically, from the true
  position and velocity (on the robot: the state estimate).
- AS-DT1: low (3 m apex) hops onto 9-16 deg slopes still land up to 12 deg
  off (~16 fits). Lidar self-occlusion (legs, foot) is not modelled; the
  AS-DT1's real range on sunlit regolith is unknown (modelled 20 m).
- Sensor realism: IMU noise/bias/latency, attitude estimator, control rate.
- Hardware prep: measure bearing friction; real cube mass (3.3 kg is a
  placeholder) and where the springs sit (their inertia is not in J_c); port
  the LQR (torque commands map to ODrive torque mode) + gating,
  landing reference and hop reference to the robot.
- Recovery success rates in README were measured on the old 6 kg model.
