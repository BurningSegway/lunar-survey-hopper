#include "rw_controller.h"

#include <cmath>
#include <iostream>

int main() {
    const ReactionWheelTuning tuning;
    WheelRestController rest_controller(tuning);
    rest_controller.begin({10.0f, -10.0f, 0.0f});

    const WheelSpeeds initial_target = rest_controller.target_speeds();
    const WheelTorques rest_torques = rest_controller.compute_motor_torques(
        {10.0f, -10.0f, 0.0f}, 0.1f);
    const WheelSpeeds ramped_target = rest_controller.target_speeds();
    const float max_rest_torque = tuning.wheel_inertia * tuning.robot_inertia
                                / (tuning.wheel_inertia + tuning.robot_inertia)
                                * tuning.wheel_rest_deceleration;
    if (!(ramped_target.x < initial_target.x) || !(ramped_target.x > 0.0f)
        || !(ramped_target.y > initial_target.y) || !(ramped_target.y < 0.0f)
        || !(rest_torques.x < 0.0f) || !(rest_torques.y > 0.0f)
        || std::abs(rest_torques.x) > max_rest_torque
        || std::abs(rest_torques.y) > max_rest_torque) {
        std::cerr << "Wheel rest should ramp targets toward zero with bounded braking torque"
                  << std::endl;
        return 1;
    }

    for (int i = 0; i < 20; ++i) {
        rest_controller.compute_motor_torques({0.0f, 0.0f, 0.0f}, 0.1f);
    }
    const WheelSpeeds stopped_target = rest_controller.target_speeds();
    if (stopped_target.x != 0.0f || stopped_target.y != 0.0f || stopped_target.z != 0.0f) {
        std::cerr << "Wheel rest targets should reach zero" << std::endl;
        return 1;
    }

    rest_controller.cancel();
    const WheelTorques cancelled_torques = rest_controller.compute_motor_torques(
        {10.0f, -5.0f, 1.0f}, 0.1f);
    if (cancelled_torques.x != 0.0f || cancelled_torques.y != 0.0f
        || cancelled_torques.z != 0.0f) {
        std::cerr << "Cancelled wheel rest should produce no braking torque" << std::endl;
        return 1;
    }

    return 0;
}
