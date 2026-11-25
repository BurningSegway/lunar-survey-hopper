# Sim_MuJoCo: notes for Claude

MuJoCo simulation of a small lunar survey hopper. It jumps with a wound-up
spring and controls its attitude in flight with three reaction wheels. The aim
is to design and test mid-flight attitude control (and, later, port it to the
real robot). `README.md` is the user-facing reference and is kept up to date;
read it for options, controls and measured results.

## Commands

```sh
.venv/bin/python -m jumper_sim --controller lqr          # viewer (default controller is pid)
.venv/bin/python -m unittest discover -s tests           # ~165 tests, ~90 s, all must pass
.venv/bin/python -m benchmarks.lqr_benchmark --label NAME      # free flight
.venv/bin/python -m benchmarks.landing_benchmark --label NAME  # hops thrown onto slopes
.venv/bin/python -m benchmarks.contact_benchmark --label NAME  # bounces + a hop
.venv/bin/python -m benchmarks.framerate_benchmark --label NAME  # viewer loop fps over a hop (no window)
.venv/bin/python -m model_comparison --label NAME           # zero g: MuJoCo vs (4.15) (+ --matlab CSV of a Simulink run), CSV + plots; --view opens a viewer
pip --python .venv/bin/python install -r requirements.txt  # venv has no pip of its own
```

Headless rendering, if ever needed: `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`.
Model tests need the generated `build/terrain_height.bin`
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
- Keep all three benchmarks (the user chose to keep them). The landing
  plane's Kalman filter (with its outlier gate) was removed on the user's
  request: the arc is fixed at lift-off and the map keeps the whole jump,
  so the newest fit at the spot is the plane.
- The LQR is the project report's (ch. 4-5, user's
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
- Design values (CAD J_c, J_w,i = 0.0044 (the CAD 0.004422 rounded, as in
  the user's fixed Simulink model; user's choice, 2026-11-20; the MATLAB
  scripts still print their gains with 0.004422, the gain test checks
  both), 3.3 kg cube placeholder, four fjedre.dk 13760 jump springs) live
  in `cube_model.py`; the MJCF is fitted to them by
  `build/fit_robot_model.py` (rerun it, don't hand-edit those MJCF values).

## Layout (`jumper_sim/`)

| File | Role |
|---|---|
| `cube_model.py` | Report ch. 4: `SENSOR_CUBE` (CAD J_c, J_w,i, mass), `JUMP_SPRING` x4, `ROBOT_MASS`; gyrostat `dynamics` f(x, u) (scalar-last quaternion), `jacobians` (4.18-4.19), `linearize` (reduced A_r, B_r at Omega_hat), `controllability_matrix`, `wheel_accelerations`. Input u = wheel motor torque tau: (J_c - J_w) w_dot = -tau - S(w) H (user's choice; replaced Omega_dot). `WheelMotor`/`WHEEL_MOTOR`: the bench-measured wheel motor (torque limit 0.43 N*m placeholder, torque falling with speed above 5.13 rev/s to 0 at 18.34 rev/s while speeding up, no lag), used only by the zero-g comparison's `--motor-limits` (its default is ideal motors, user's choice: the report figures compare the models without actuator dynamics). `report_dynamics` is (4.15) as written (10 states, its tau = J_w Omega_dot, J_c^-1), `motor_torque` converts; `model_comparison/compare.py` checks MuJoCo against it in zero g (bearing friction off, foot settled into its limit first) |
| `controller.py` | PID and the report's LQR (ch. 5) as the script's LQI: u = tau = -K_bar [q_err_vec, w - w_ref, x_I] (torque straight to the motors, clipped at the motor limit), `augment`, `tracking_feedforward`, CARE by matrix sign function, Bryson's rule + anti-windup via `LqrLimits`, fixed gain at Omega_hat = 0 (optional rescheduling at measured wheel speeds); `reference_rate`, bearing-friction feedforward, wheel limits, wheel-rest ramp, `robot_inertia` (at qpos0 = the flight pose) |
| `simulation.py` | `JumperSimulation`: one physics step (`mj_step`, 1 ms) with the attitude controller every 10th (100 Hz, torques held); contact gating on the IMU estimate (`estimated_airborne`, `_set_airborne`: attitude control from its lift-off to its impact, wheel momentum bleed otherwise, wheel torques zero from the jump command to the cutoff; the estimate skips the first 10 ms after a start/reset, `IMU_START_SETTLE_TIME`); the same estimate drives the phases, the tracker's restarts (`touchdown` on every IMU-ground sample, before the sample is integrated), lidar, scan, landing fits, touchdown record, height label and disc colour, and jump-aim legs straighten at its lift-off (still in TAKEOFF); sim contact `in_contact` (60 ms debounce, `contact_phases`) is ground truth for tests/benchmarks only (a test blinds it and gets a bit-identical hop); jump sequence; wires lidar, flight phase, hop reference and scanning together; `LidarLandingReference` (lidar batches -> ground map, fits every 50 ms) |
| `flight_phase.py` | Hop phase state machine GROUND / TAKEOFF / ASCENT / DESCENT / LANDING (jump command, 0.5 m above stance from integrated IMU vertical velocity, apex, touchdown near (landing plane's near-touchdown rule) or contact, still 0.5 s, the last with no contact condition); contact = the IMU's estimate; all behaviours key off it. Starts in DESCENT |
| `flight_reference.py` | Hop reference: TAKEOFF holds the attitude `start_flight` gets, on a jump the one at the jump command (`_jump_attitude` in `simulation.py`, before the push turned the body; user's request, 2026-11-24; measured with control starting 8 ms after lift-off, emulating the cutoff: the old hold was the attitude at control start, 1.1-1.4 deg off the pre-jump one; since the IMU gate (2026-11-25) control starts at the cutoff, in TAKEOFF, so the hold is used), on a bounce the lift-off attitude; on a jump's ascent body -z points at a settable world target, tilt cap 60 deg, and on into the descent for half the ascent's time (user's stand-in for a landing-spot criterion); then back to level, then the landing plane, updated until impact. Heading: the yaw reference (lift-off heading) throughout (face-travel went with the AS-DT1; `--spin` was removed for now, user's choice). Slew limit 180 deg/s |
| `ray_casting.py` | Exact fast heightfield ray caster (matches `mj_multiRay`; skips 8x8-cell blocks the ray passes above their highest vertex, block looked up at the skipped segment's middle: ~5x fewer march steps, 2.4-3x faster; its cost is set by the number of march steps, i.e. ray length, more than by ray count) and a mesh ray caster for objects on it (einsum, not `@`: OpenBLAS threading made it 2-3x slower); site/geom names. The RealSense D435 depth camera that used to live here was removed (lidar only) |
| `landing_reference.py` | `GroundMap` of the jump's lidar points (0.5 m grid, emptied at each jump; only cells within `map_corridor` = 4 m (user's choice, margin: reads are within ~1.5 m) of the arc's ground track are binned, the rest wait unbinned in 8 m tiles and are binned at the first read after a new (bounce) arc, from the tiles near it only (`set_track`, fed by `LidarLandingReference.set_arc`; a settling robot latches many short arcs, binning at each latch froze the loop for 2 s); reads identical to binning all, tested; airborne viewer loop 37 -> 58 fps; integer cell keys `i * _CELL_RADIX + j`), ballistic landing-spot prediction along the tracker's arc (base within 0.3 m of the mapped ground), area-weighted plane fit of <= 1000 points within 0.5 m of the spot, calibrated uncertainty (noise vs. roughness), `LandingPlane` = the newest fit (ignored if its noise sigma > 3 deg; the Kalman filter was removed, user's choice); keeps updating until impact (user's choice; `freeze_near_touchdown` restores the old freeze), its near-touchdown rule starts LANDING. Benchmarks/tests throw the robot onto a spot (starting in ASCENT with the IMU vertical velocity set and the arc latched from the start state) |
| `ground_contact.py` | `GroundContactEstimator`: airborne or not from the IMU's specific force alone (in units of local g): airborne at the jump's cutoff (>= 1.25 g, then < 0.5 g, the tracker's thresholds) in TAKEOFF, no delay; an impact (> 1.25 g) ends it in DESCENT/LANDING only (the spring wind-back reads ~1250 g in the ascent); bounces after < 0.5 g for 60 ms unbroken (`FREE_FALL_DELAY`, shared with the tracker's latch delay); phase GROUND ends a flight it still holds (missed soft landing). The robot's only contact sense (see Open items) |
| `trajectory.py` | `TrajectoryTracker`, the firmware's tracker (main, `teensy-lqr-stabilizer` "trajectory estimation"): from the jump command, Euler-integrates the IMU's world linear acceleration from rest; cutoff (`cutoff_time`, `cut_off`) when the specific force drops < 0.5 g after > 1.25 g (~40 ms after release), keeps integrating and latches p, v0 (`arc_time`, `arc_position`, `arc_velocity`) `latch_delay` = `FREE_FALL_DELAY` (60 ms, `ground_contact.py`, shared with the bounce delay; user's choice) later; then a fixed ballistic arc until touchdown; `touchdown` (every contact step) restarts integration from the arc's state there, so each bounce's arc latches 60 ms after its cutoff (user's request); `latch` for thrown/dropped starts; `path` samples an arc. No arc, no landing fits. Arcs >= 60 ms of tracked jumps go to `<name>.arcs.json` with the true base path sampled on the same 20 ms grid (`_record_arc`, `_sample_actual_path`) |
| `jump_aim.py` | On the ground, lean the body axis towards the track target (3 deg near, 8 deg at >= 30 m), closed loop on IMU tilt via a leg-shortening plane (integrator, only while still, held during release, reset at touchdown, legs capped at 10 cm, off on ground > 20 deg); motors `tilt_fl/fr/bl/br` |
| (target, tipped) | `simulation.py`: bunny `target_object` placed on the terrain at `--track-target` (offset from `mesh_pos`; parked 100 m down otherwise), seen by `MeshRayCaster` in the terrain caster's `objects` (`JumperSimulation._terrain`); tipped (> 60 deg in GROUND) pulls all legs in, and Space then refuses to jump: R resets (user's choice; the in-air recovery, `recovery.py`, was removed). Righting on the ground is NOT possible (tested: wheel push, wheel spin-brake, wheel rock, leg kick incl. 1000 N push) |
| `pointcloud.py` | Lidar point cloud: every point of each tracked jump, lift-off to touchdown (per-point `range`), binary PLY at `pointclouds/target_scan.ply`, emptied at program start, rewritten at each touchdown (~21 MB per full-rate hop); touchdowns in `<name>.touchdowns.json`, predicted paths (jump + bounces) in `<name>.arcs.json` |
| `lidar.py` | The only sensor: Livox Mid-360S on site `lidar`: +x side face, flush with the bottom, axis out of the face tilted 22.5 deg down (MJCF `euler="0 -67.5 0"`), so body -z is mid-band (band 38-97 deg off axis; blind cone around the axis); 2 cm range noise, 4 cm within 1 m; no self-occlusion. A bottom mount facing down (blind below, tracked 45 deg off the target) was tried and replaced: vertical drops tipped; Livox's recorded pattern from `mujoco-lidar` (R2 fill without it; it sweeps its elevation band in ~50 ms, so test over >= 50 ms), cast in 10 ms batches while airborne at `point_fraction` (1.0 with a GPU, 0.1 without). The user chose it over the Avia, the RealSense and the Sony AS-DT1 (bottom, 24x24 grid; removed with its `--lidar` option and face-travel heading), which were all removed |
| `gpu_ray_caster.py` | `WarpRayCaster`: the heightfield caster's triangles + objects on the GPU (NVIDIA Warp BVH), identical hits, ~80x faster (`gpu_available()`; numpy fallback). Warp's first init runs `uname -p` via Popen (tests that mock Popen must filter) |
| `cloud_viewer.py` | Live Open3D window watching that PLY, `<name>.touchdowns.json` (green labelled spheres), `<name>.arcs.json` (predicted paths, lines orange -> red by jump, with the true base path over the same time in magenta and labels of their distance at the apex and touchdown, `path_errors` in `pointcloud.py`); points coloured blue -> green by jump (both spectra from jump 1 to the latest, user's choice: paths must stand out from the points); points within 0.3 m of the target white and larger (own process, started by `app.py`); GUI `SceneWidget`, not the classic Visualizer (one global point size) nor `O3DVisualizer` (no key callbacks in Python) |
| `visualization.py` | Overlays in the MuJoCo viewer: landing plane at the predicted spot, target, scan sample, `h = ...` label (base height above the last ground contact; MuJoCo 3.2.3 has no `set_texts`, so it is an `mjGEOM_LABEL`) |
| `app.py` | CLI and viewer loop (`--smooth`: Mid-360S at 3 % of its point rate, measured to run without stutter on the CPU): sim paced to the wall clock, viewer synced at 60 Hz (user's choice: no need for more); `RobotFollower` centres a free camera on the robot each sync (MuJoCo's tracking camera moves 20 % per *drawn* frame from the viewer's own thread: it lagged and made a fast robot stutter) |
| `telemetry.py`, `status_indicator.py` | PlotJuggler UDP stream (127.0.0.1:9870; layout in `PlotJuggler_Setup.xml`), rest indicator |

Model: `build/Jumping_Robot.xml`. Lidar site `lidar` under the base;
terrain = moon crater field from `build/Testing/Temporary/moon-surface`
(100 x 100 m, cm -> m, Y-up -> Z-up) made by `build/convert_terrain.py`
(renamed from `convert_usd.py`; USD loader, loose-rock removal, OBJ and
texture output dropped, user's choice; output byte-identical), centred with
the spawn ground at z=0; one heightfield, no visual mesh (user's choice: the
viewer draws what is simulated), in geom group 2 (drawn by default) with firm contact
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

- Replace simulator contact with IMU sensing (user's request): done.
  Step 1 done (2026-11-19): `ground_contact.py`'s
  `GroundContactEstimator` (`estimated_airborne`, updated in
  `_update_flight_phase` from the phase before the sample). Step 2 done
  (2026-11-25): attitude control is gated on it (`_set_airborne`): on at
  the cutoff (`_lift_off`: flight mode, yaw latch, `start_flight`), off at
  the impact (bleed); `reference_quaternion` and the flight reference's
  update follow it; the jump command cancels the bleed (wheel torques 0
  until the cutoff; a jump that fails restarts it in GROUND). The estimate
  skips the first 10 ms after a start/reset (user's choice over settling
  the start state): the wound foot settling onto its stop jolts the base
  26-31 g for 2 samples (> 1.25 g up to 4 ms), read as an impact it
  switched control off for 65 ms and broke the free-flight benchmark; the
  tracker and phases still get those samples. Measured: control 1 ms after
  the cutoff (was 59-60 ms), 0-1 ms after the true lift-off, but 2 ms
  before it on the Earth hop to (10, 5) (foot touching lightly at < 0.5 g);
  apex errors and wheel speeds unchanged (within 0.3 deg / 0.05 rev/s);
  lqr and landing benchmarks identical; contact benchmark 5 of 26 runs
  differ, spawn hop touchdown 0.68 -> 0.72 deg, and the 3 m drop at
  (6.6, -6.9) (tips over before and after) peaks 3.3 -> 7.0 rev/s: while
  toppling over a leg edge (tilt > 78 deg) |f| < 0.5 g for 60 ms reads as
  a bounce and control runs in contact. Step 3 done (2026-11-25): the
  phases (contact input = the estimate; LANDING -> GROUND stillness without
  a contact condition, and GROUND ends a flight the estimate still holds),
  the tracker's restarts, lidar, scan open/close, touchdown record, landing
  fits, height label and disc colour all use the estimate; sim contact is
  ground truth only (`test_nothing_the_robot_does_uses_the_simulator_contact`
  blinds it: bit-identical hop). ASCENT now starts at the 0.5 m mark (84 ms
  after the jump command on the Moon hop to (10, 5), was 101 ms behind the
  60 ms debounce), which released the jump-aim legs 19 ms before the arc's
  latch and put their rebound in the arc (Moon jump touchdown 11.3 ->
  14.6 cm); the legs now straighten at the IMU's lift-off instead (user's
  choice): 11.2 cm. Measured against the arc-delay commit (CPU caster):
  arcs Moon jump 11.3 -> 11.2 cm, bounces 16.4 -> 16.0, Earth 2.9 -> 3.6
  (arcs now end at the IMU's impact, 1-6 ms into the contact); lqr and
  landing benchmarks identical; take-off the same on the Moon, Earth apex
  errors 7.4 -> 6.4 / 8.9 -> 8.6 deg; contact benchmark (gated) drops'
  touchdowns identical, spawn hop 0.45 -> 0.57 deg, no tip-overs, wheels
  end at 0. Known limits: impacts count only in DESCENT/LANDING (a hit in
  ASCENT goes unseen), and toppling over a leg edge reads as a bounce.
  Shadow check vs sim contact (12 Moon drops from 3 and 6 m onto the
  landing benchmark's spots + the spawn hop, Moon and Earth, 2026-11-19):
  every jump lift-off and nearly every touchdown on the true sample;
  bounces on 52-61 ms after the true lift-off; airborne while touching:
  1-6 ms at most touchdowns (force not yet > 1.25 g), up to ~60 ms on soft
  touches below 1 g after a last small bounce, one switch-on in contact
  (58 ms, < 0.3 g, drop 6 m spot 0):
  - Lift-off -> control on, at the tracker's cutoff, with no delay
    (user's decision, 2026-11-19; replaces the earlier fixed-delay plan).
    The cutoff is the first IMU sample on which the IMU no longer sees the
    spring pushing the robot: `TrajectoryTracker.update`, per sample, sets
    "launched" once the specific-force magnitude |f| reaches 1.25 g (the
    push has started), and on the first later sample with |f| < 0.5 g
    latches the arc (its time, p, v0); g is the local gravity, so 0.5 g is
    0.81 m/s^2 on the Moon and 4.9 m/s^2 on Earth. Why that means "off
    the ground": an accelerometer reads every force on the robot except
    gravity, and the only one besides gravity is the ground's push through
    the foot (and legs). It is ~40-210 g during the push, and drops to ~0
    when nothing touches the ground any more, i.e. free fall. So the cutoff
    marks the end of the ground's force, which is also all that could
    react the wheel torque. It relies on the sim's IMU being unclipped (see
    the IMU noise item). The same sample is the first trajectory estimate
    (the arc is latched there), so control starts and the arc exists at
    the same sample; at 100 Hz control, the first control update after it.
    Measured (default hop, Moon and Earth, 2026-11-19; sim contact as
    truth): the cutoff falls 0-2 ms after the true lift-off (39-40 ms
    after the jump command; the legs leave at 2-3 ms, the foot touches at
    9 ms), on the jump and on every bounce; today's gating (60 ms debounce)
    starts control at cutoff + 60 ms. In the sim the IMU is the velocity
    change over the last 1 ms step, so it lags the contact by up to a step.
    Kept as is (user's choice): the yaw reference latch and the start of
    target tracking (`flight_reference.start_flight`) happen when the
    controller starts, now the cutoff. No wheel momentum bleed through
    the push (user's choice, for now): wheel torques are zero from the jump
    command until the controller starts. ASCENT keeps its 0.5 m marker.
    Bounces re-engage control after |f| < 0.5 g for 60 ms without a break
    (user's choice (b), 2026-11-19, "for now"): on the ground |f| also dips
    below 0.5 g while the robot rocks or slides, mostly for < 20 ms (up to
    74 and 139 ms in light contact, Earth hop and a 6 m drop); measured
    offline over the drops and hops, 60 ms (today's `FLIGHT_DEBOUNCE`)
    catches all 23 bounces > 0.1 s, 60/123 ms median/max after the true
    lift-off, 2 switch-ons in contact; 30 ms: 2 too; 100 ms: 1; 150 ms: 0
    but one bounce missed. Test with
    sim contact as truth: contact at control start (foot, legs, body),
    wheel speeds at control start and at the apex, target pointing on the
    ascent, the landing benchmark's touchdown attitude, Moon and Earth.
  - Landing -> control off: the prediction only arms (landing plane's
    near-touchdown rule: < 1 m below or < 0.3 s away; the lidar's distance
    to the plane beats the arc's height, which is fixed at cutoff and
    drifts), the IMU impact spike switches the flight controller off.
    Switching off on the prediction alone would lose the final alignment
    with the plane. Impacts count only in DESCENT and LANDING (user's
    choice, 2026-11-19: no need to handle impacts outside the landing; in
    DESCENT too so a landing the lidar never armed still ends the flight;
    |f| > 1.25 g). A spike alone is not
    a landing: the spring wind-back at 0.4 s reads up to ~1250 g in
    flight on the Moon (~25 g for most of 400-428 ms; 4-9 g on Earth;
    measured 2026-11-19, the peak is likely the stiff stroke limit), while
    an impact reads 8-15 g on its first contact sample.
  - Stationary -> GROUND: the existing stillness rule (|f| - g < 0.5 m/s^2,
    rate < 0.2 rad/s for 0.5 s) without the contact condition.
  Measure each change (take-off wheel speeds, tracking, landing benchmark).
- Freeze the arc later than the cutoff: done (2026-11-25). The tracker
  latches `FREE_FALL_DELAY` (60 ms, one constant with the contact
  estimate's `bounce_free_fall_time`, user's choice) after the cutoff, for
  jumps and bounces; the cutoff stays the first sample < 0.5 g and is not
  re-armed by the post-cutoff push (~3 g). Measured in the sim (six hops,
  targets (10, 5), (-6, 12), (15, 0), mean, arc vs true base path):
  Moon jump apex 24.3 -> 6.4 cm, touchdown 50.0 -> 11.3 cm; Moon bounces
  (6) touchdown 60.7 -> 16.4 cm; Earth jump touchdown 6.8 -> 2.9 cm
  (matches the offline replay). Landing benchmark identical (it latches its
  own start state); contact benchmark: first touchdowns unchanged, spawn
  hop (gated) 0.82 -> 0.45 deg. On the robot the size of the post-cutoff
  push depends on the real foot and springs (and the +-10 g clipping), so
  re-check the delay there.
- Tip catch at landing (user's idea, ON HOLD: don't implement yet). After
  impact, if the robot is about to topple over a leg edge, spend the wheels'
  full momentum to stop it. Its own law (an inverted pendulum about the
  edge, not the free-body flight LQR): brake the tipping rate at full motor
  torque. Trigger on energy: tips if rotational energy about the edge >
  energy to lift the CoM over the edge; catch only if the wheels' remaining
  momentum covers the difference, else accept the fall. Inputs: IMU tilt and
  rate, landing plane normal. Rough numbers: inertia about an edge ~0.52
  kg*m^2 (leg tips +-0.195 m, 0.26 m below the CoM), so ~0.45 N*m*s per
  wheel absorbs ~0.9 rad/s; static tip angle ~37 deg on flat ground (minus
  the slope downhill); on the Moon with 10 deg to the edge it tips above
  ~0.35 rad/s. Only before the CoM passes the edge (righting from the
  ground does not work, tested), and only with the wheels near rest at
  touchdown (take-off spin stays stored in them; > ~4 rad/s leaves none).
  Fixes the open item below (tipping after touchdown on steep spots).

- Robot tips over after touchdown on steep spots (contact benchmark), incl.
  the spawn hop's landing; attitude control is gated off in contact.
- Point-cloud processing (range crop, voxel downsample, registering jumps).
- Landing-spot prediction (could become the default track target). The
  landing reference already predicts it along the IMU tracker's arc.
- Bounce prediction (user's idea, for later): the landing plane fitted at
  the predicted spot, with the arc's incoming velocity, could give a rough
  estimate of the bounce's trajectory before touchdown (e.g. to bin the
  ground map's corridor for it early). Now a bounce's arc is only known
  once the tracker latches it on leaving the ground.
- Lidar self-occlusion (legs, foot) is not modelled.
- IMU noise (user's request, not done yet): simulate the robot's Xsens
  MTi-610 (leaflet 2025-02-07, typical values). Accelerometer: +-10 g range,
  60 ug/sqrt(Hz) noise, bias stability 10 ug (x, y) / 15 ug (z), 500 Hz
  bandwidth. Gyro: +-2000 deg/s, 0.007 deg/s/sqrt(Hz), bias stability
  8 deg/h, 520 Hz bandwidth, g-sensitivity 0.1 deg/s/g. Output up to 2 kHz
  (400 Hz with its strapdown integration). At 1 kHz that is ~0.013 m/s^2
  and ~0.16 deg/s per sample (density x sqrt(500 Hz)): far below the Moon's
  1.62 m/s^2 standing vs ~0 in free fall, so IMU phase detection is not
  noise-limited; the biases barely matter over a hop. Magnetometer and
  barometer are useless on the Moon (no field to speak of, no air).
  **Not clipped in the sim (user's decision, 2026-11-19)**: the sim's IMU
  reads any specific force, but the real MTi-610 saturates at +-10 g.
  Measured in the sim (Moon hop, finite-difference IMU, 2026-11-19), the
  spring push peaks at ~210 g and is > 10 g for 27 ms, the landing ~208 g
  and > 10 g for 48 ms (the firm contact makes sharp spikes; even the
  springs' 1718 N on ~4.2 kg is ~42 g). So on the robot the tracker's
  integration of the push (its v0, hence the whole arc) would come out too
  low: before the port, measure the v0/arc error with clipping, and if
  needed get v0 another way (spring model, or the first lidar fits of the
  ground). Clipping does not affect the timing: the cutoff (a drop below
  0.5 g) and the impact spike detect the same when clipped. Also still open:
  latency and an attitude estimator (the sim uses the true attitude).
- Control rate: 100 Hz done (2026-11-25, `CONTROL_PERIOD`,
  `control_period=`): the wheel block of `_apply_controls` runs every 10th
  physics step (1 ms physics kept) and the torque is held (ZOH); the
  integrators, `reference_rate`, `WheelRestController` and
  `limit_wheel_command` (new `time_step` argument) take the control
  period; the jump command zeroes the held wheel torque at once (no bleed
  through the push). Gain kept continuous (the script's printed gains are
  the CARE solution; `dlqr` at 0.01 s gives 744 / 123 / 692, 3-4 % lower;
  fastest closed-loop pole 3.7 rad/s, the ZOH loop's largest |z| 0.984).
  `control_period=0.001` reproduces the 1 kHz code exactly (checked on
  the contact and lqr benchmarks). Measured 1 kHz -> 100 Hz on the CPU
  caster (the GPU was unavailable): lqr the same (IAE -1..+2 %); landing
  median 0.39 -> 0.46, p90 0.99 -> 1.43 deg (3 m better, 6/10 m worse,
  two 6 m hops 0.9 -> 2.3 and 1.0 -> 1.8 deg); contact (gated) touchdown
  median 0.36 -> 0.33 deg, tip-overs 2 -> 0; ungated wheels now stop at
  19.53 rev/s (97.7 %, just under the benchmark's 98 % "saturated"
  threshold, so its sat% reads 0). Take-off: apex errors and wheel speeds
  within 0.6 deg / 0.15 rev/s. Still to decide with the user: the rate of
  the IMU-side parts (tracker, contact estimate, phases; the MTi-610 gives
  400 Hz-2 kHz) and of the hop reference (`flight_reference.update`), all
  still every 1 ms step. Note `RW_PID.cpp`'s `pumpEvents` delays each
  loop 10 ms on the MCP2515, so the robot cannot run faster than ~100 Hz.
- Wheels in the sim vs the robot (for the port): the wheels are propagated
  by MuJoCo only (hinge joints driven by ideal torque motors, viscous
  bearing friction = joint damping 0.01 N*m*s/rad, gyroscopic coupling from
  the multibody dynamics); the controller never integrates a wheel model, it
  only reads the true wheel speeds (`qvel`) for the speed limit, the
  friction feedforward (LQR), the wheel-rest/bleed controller and the
  optional gain scheduling (the LQI's state has no wheel speeds). On the
  robot those speeds come from the ODrives' encoder estimates over CAN
  (filtered, delayed, at the CAN rate), and the torque is not ideal (torque
  constant, back-EMF limiting torque near top speed, Coulomb friction and
  cogging the sim's viscous damping does not have). Bench test
  (2026-11-20, `cube_model.WHEEL_MOTOR`, placeholders until the user's test
  setup on the robot, the same for all three wheels (user's choice); the
  findings table is in README "Wheel motors"): no measurable lag (current
  at its setpoint within one ~20 ms log sample; user: lag is what matters
  most, so none modelled); torque falls linearly above 5.13 rev/s to 0 at
  18.34 rev/s while speeding the wheel up (top speed 16.4 rev/s against
  friction); 0.43 N*m is the test's command, not a measured limit; friction
  0.03-0.07 N*m (not modelled, user's choice); Kt/J measured 57.6
  rad/s^2/A, i.e. J_w 0.0023 at the ODrive's Kt 0.13 N*m/A, but the user
  keeps 0.0044 in the sim (the wheels differ). The 0.3 N*m doublet demo
  plot was dropped (user: test it on the robot). Only the zero-g
  comparison's `--motor-limits` uses it (default ideal, user's choice); the full sim still has the made-up 20 rev/s / 1.95 N*m
  limits (`ReactionWheelTuning`) and the MJCF's +-20 N*m.
- Wheel mass placeholder: `SENSOR_CUBE.wheel_mass` (0.2 kg, made up) in
  `cube_model.py`, written to the MJCF by the fit script (user's choice,
  2026-11-24: a named placeholder until the real mass is known; MuJoCo
  rejects 0 for a moving body). It moves the CoM ~6 mm along +x, +y, +z
  (wheels 0.125 m off-centre), so the real value changes the take-off push;
  total mass and J_c are unaffected.
- Lidar cut-off range: change `LIVOX_MID360.max_range` from 40 m to 100 m
  (user's request, not done yet). 40 m is the datasheet's typical range at
  10 % reflectivity (about regolith's); 100 m is where the sensor cuts off.
  Update the `lidar.py` docstring and the README lidar table/text (40 m,
  ~line 403-406), and measure the casting cost before/after: the casters'
  cost grows with ray length (march steps), though the 100 x 100 m terrain
  ends rays sooner anyway. Range noise stays the two datasheet cases (2 cm,
  4 cm within 1 m; user's choice, no further range dependence).
- Hardware prep: measure bearing friction; real cube mass (3.3 kg is a
  placeholder) and where the springs sit (their inertia is not in J_c); port
  the LQR (torque commands map to ODrive torque mode) + gating,
  landing reference and hop reference to the robot.
