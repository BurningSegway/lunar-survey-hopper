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
- `C`: request/cancel controlled wheel rest; rest ramps the wheels toward zero
  while holding the current yaw reference
- `X`: disable reaction-wheel motor torque; press `C` to re-enable stabilization
- `R`: reset the simulation and restore Earth gravity

MuJoCo's viewer has its own Backspace reset shortcut. The simulator restores
the previous physics state when that default action occurs, then applies the
gravity toggle; use `R` when an actual reset is desired.

Telemetry is sent as JSON UDP datagrams to `127.0.0.1:9870` using the
PlotJuggler signal names from the existing setup file.

## Tuning

Edit `ReactionWheelTuning`, `PidGains`, and `LqrWeights` in
`jumper_sim/controller.py` to change the PID gains, wheel speed/acceleration
limits, wheel-rest behavior, mass/inertia values, and LQR weights. Keep the
robot and wheel inertia values consistent with `build/Jumping_Robot.xml`.

## Tests

Run controller, telemetry, and available headless MuJoCo tests with:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

The headless model tests require the locally generated
`build/terrain.obj` and `build/terrain_height.png` and are skipped with a setup
hint when either file is absent.
