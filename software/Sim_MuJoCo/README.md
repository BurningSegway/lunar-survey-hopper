# MuJoCo simulator

The simulator uses the MuJoCo distribution in `mujoco-3.2.3/` and the robot
model at `build/Jumping_Robot.xml`. Keep the working directory at this folder
when running the executable so the model can be found.

## Build and test

Requirements: CMake 3.10+, a C++17 compiler, GLFW 3, and OpenGL development
libraries. On Linux, configure and build out of tree:

```sh
cmake -S . -B cmake-build
cmake --build cmake-build
ctest --test-dir cmake-build --output-on-failure
```

Run from this folder:

```sh
./cmake-build/jumper_sim --controller pid --gravity earth
```

Options are `--controller pid|lqr` and `--gravity earth|moon`. Backspace
switches Earth/Moon gravity during the simulation; reset (`R`) restores the
gravity selected at launch.

## Tunable parameters

Edit the defaults in `rw_controller.h` in `ReactionWheelTuning` to adjust the
wheel speed/acceleration limits, wheel-rest behavior, robot and wheel mass and
inertia, and per-axis PID gains. `LqrWeights` in the same file holds the LQR
angle, angular-rate, and torque weights. Keep the mass and inertia values
consistent with the inertial properties in `build/Jumping_Robot.xml`.

## Controls

- `Space`: initiate a jump
- `Enter`: toggle directional jump tilt
- `Backspace`: toggle Earth/Moon gravity
- `C`: request or cancel controlled wheel rest; rest holds the current yaw
- `X`: disable/zero reaction-wheel motor torque
- `R`: reset the simulation
