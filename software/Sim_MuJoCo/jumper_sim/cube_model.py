"""The sensor cube's model (report chapter 4): a gyrostat with three reaction wheels.

State ``x = [q1 q2 q3 q4 wx wy wz]``: the attitude quaternion of the body frame
in an inertial frame, scalar last (``q4``), and the body angular velocity in
the body frame. ``Omega`` are the wheel speeds relative to the body, one wheel
per body axis.

The report writes ``u = Omega``, but the input matrix it derives (4.19) is the
derivative with respect to the wheel *acceleration* ``Omega_dot``: ``Omega``
itself only enters ``A`` (4.21), as the operating point ``Omega_hat``. So the
linear model's input is ``Omega_dot``, and the wheel speed command sent to the
motors is its integral (see :class:`jumper_sim.controller.LqrController`).

In MuJoCo's convention (scalar first, Hamilton, body to world), the report's
quaternion is ``(x, y, z, w)``: (4.3)-(4.4) is ``q_dot = 1/2 q * (0, w_body)``
written out with the scalar last.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The report's linearisation point (5.1): q0 = [0 0 0 1], w0 = 0.
EQUILIBRIUM = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


@dataclass(frozen=True)
class SensorCube:
    """Mass properties of the sensor cube.

    ``body_inertia`` is the diagonal of J_c: the whole robot about its centre
    of mass in the body frame, with the wheels locked to it (the gyrostat's
    H = J_c*w + J_w*Omega counts the wheels' spin on top of that).
    ``wheel_inertia`` is J_w,i, each wheel about its spin axis.
    """

    body_inertia: tuple[float, float, float]  # J_c,x, J_c,y, J_c,z [kg*m^2]
    wheel_inertia: float  # J_w,i [kg*m^2]
    mass: float  # [kg]

    @property
    def j_c(self) -> np.ndarray:
        return np.diag(self.body_inertia)

    @property
    def j_w(self) -> np.ndarray:
        """(4.6): the three wheels are identical."""
        return self.wheel_inertia * np.eye(3)


# The final cube design, from CAD (inertias given in g*mm^2 -> 1e-9 kg*m^2).
# The mass is a placeholder until the design's is known.
# build/Jumping_Robot.xml is fitted to these values by build/fit_robot_model.py;
# rerun it after changing them.
SENSOR_CUBE = SensorCube(
    body_inertia=(70660000e-9, 70670000e-9, 73330000e-9),
    wheel_inertia=4.422e6 * 1e-9,
    mass=3.3,
)


@dataclass(frozen=True)
class JumpSpring:
    """One compression spring of the jump mechanism (datasheet values)."""

    stiffness: float  # [N/m]
    stroke: float  # free length to max. loaded length [m]
    mass: float  # [kg]

    @property
    def max_force(self) -> float:
        return self.stiffness * self.stroke

    @property
    def energy(self) -> float:
        return 0.5 * self.stiffness * self.stroke**2


# fjedre.dk 13760, piano wire, d 4 mm, De 54 mm, L0 335 mm, Ln 77.8 mm,
# R 1.67 N/mm, 224 g: 55.2 J wound to Ln. Four in parallel push the foot, wound
# to full compression before each jump.
JUMP_SPRING = JumpSpring(stiffness=1670.0, stroke=0.2572, mass=0.224)
JUMP_SPRING_COUNT = 4
# What flies: the sensor cube and the springs (J_c stays the CAD cube's).
ROBOT_MASS = SENSOR_CUBE.mass + JUMP_SPRING_COUNT * JUMP_SPRING.mass


def quaternion_rate_matrix(w: np.ndarray) -> np.ndarray:
    """S_q (4.4): q_dot = 1/2 S_q q."""
    wx, wy, wz = w
    return np.array([
        [0.0, wz, -wy, wx],
        [-wz, 0.0, wx, wy],
        [wy, -wx, 0.0, wz],
        [-wx, -wy, -wz, 0.0],
    ])


def cross_matrix(v: np.ndarray) -> np.ndarray:
    """S (4.9): S(v) a = v x a."""
    vx, vy, vz = v
    return np.array([
        [0.0, -vz, vy],
        [vz, 0.0, -vx],
        [-vy, vx, 0.0],
    ])


def dynamics(
    x: np.ndarray,
    wheel_speeds: np.ndarray,
    wheel_accelerations: np.ndarray,
    cube: SensorCube = SENSOR_CUBE,
) -> np.ndarray:
    """x_dot = f(x, u) (4.11), free flight (no external torque, 4.7)."""
    q, w = x[:4], x[4:]
    momentum = cube.j_c @ w + cube.j_w @ wheel_speeds  # (4.5)
    q_dot = 0.5 * quaternion_rate_matrix(w) @ q  # (4.3)
    w_dot = np.linalg.solve(  # (4.10)
        cube.j_c, -cube.j_w @ wheel_accelerations - cross_matrix(w) @ momentum
    )
    return np.concatenate([q_dot, w_dot])


def jacobians(
    x: np.ndarray, wheel_speeds: np.ndarray, cube: SensorCube = SENSOR_CUBE
) -> tuple[np.ndarray, np.ndarray]:
    """A = df/dx (4.18) and B = df/dOmega_dot (4.19), anywhere."""
    q, w = x[:4], x[4:]
    momentum = cube.j_c @ w + cube.j_w @ wheel_speeds
    # d(1/2 S_q(w) q)/dw: column i is 1/2 S_q(e_i) q.
    dq_dw = 0.5 * np.column_stack(
        [quaternion_rate_matrix(np.eye(3)[i]) @ q for i in range(3)]
    )
    # d(w x H)/dw = -S(H) + S(w) J_c, with H = J_c w + J_w Omega.
    dw_dw = np.linalg.solve(cube.j_c, cross_matrix(momentum) - cross_matrix(w) @ cube.j_c)
    a = np.zeros((7, 7))
    a[:4, :4] = 0.5 * quaternion_rate_matrix(w)
    a[:4, 4:] = dq_dw
    a[4:, 4:] = dw_dw
    b = np.zeros((7, 3))
    b[4:, :] = -np.linalg.solve(cube.j_c, cube.j_w)
    return a, b


def linearize(
    wheel_speeds: np.ndarray, cube: SensorCube = SENSOR_CUBE
) -> tuple[np.ndarray, np.ndarray]:
    """Reduced A_r, B_r (5.1) about q0 = [0 0 0 1], w0 = 0, Omega_0 = Omega_hat.

    At q0 the row (and column) of q4 is zero: q4 = sqrt(1 - |q1..3|^2) is a
    dependent state, so it is stripped, leaving x_r = [q1 q2 q3 wx wy wz].
    """
    a, b = jacobians(EQUILIBRIUM, np.asarray(wheel_speeds, dtype=float), cube)
    keep = [0, 1, 2, 4, 5, 6]
    return a[np.ix_(keep, keep)], b[keep, :]


def controllability_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """[B AB A^2B ... A^(n-1)B] (5.2)."""
    blocks = [b]
    for _ in range(a.shape[0] - 1):
        blocks.append(a @ blocks[-1])
    return np.hstack(blocks)
