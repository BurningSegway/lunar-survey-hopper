#include "rw_controller.h"

#include <cmath>
#include <stdexcept>

LqrController::LqrController(const ReactionWheelTuning& tuning, const LqrWeights& weights)
    : tuning_(tuning), weights_(weights) {
    if (tuning_.robot_inertia <= 0.0f || weights_.angle <= 0.0f
        || weights_.angular_rate <= 0.0f || weights_.torque <= 0.0f) {
        throw std::invalid_argument("LQR inertia and weights must be positive");
    }
    gains_ = compute_axis_gains();
}

std::array<double, 2> LqrController::compute_axis_gains() const {
    // Closed-form continuous LQR gains for theta_dot=omega and omega_dot=body_torque/inertia.
    const double body_inertia = static_cast<double>(tuning_.robot_inertia);
    const double angle_weight = static_cast<double>(weights_.angle);
    const double rate_weight = static_cast<double>(weights_.angular_rate);
    const double torque_weight = static_cast<double>(weights_.torque);
    const double p_angle_rate = body_inertia * std::sqrt(torque_weight * angle_weight);
    const double p_rate = body_inertia * std::sqrt(
        torque_weight * (2.0 * p_angle_rate + rate_weight));

    return {{
        p_angle_rate / (torque_weight * body_inertia),
        p_rate / (torque_weight * body_inertia),
    }};
}

TorqueCommand LqrController::compute_torque(const SensorData& sensor, const WheelSpeeds&,
                                             float) {
    // Internal wheel torque cannot remove net angular momentum, so this local LQR regulates
    // body angle and rate while wheel-speed limits remain in the shared actuator layer.
    const auto axis_torque = [this](float angle, float angular_rate) {
        return static_cast<float>(-(gains_[0] * angle + gains_[1] * angular_rate));
    };

    return {
        axis_torque(sensor.roll, sensor.roll_rate),
        axis_torque(sensor.pitch, sensor.pitch_rate),
        axis_torque(sensor.yaw, sensor.yaw_rate),
    };
}
