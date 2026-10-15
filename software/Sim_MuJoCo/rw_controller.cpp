#include "rw_controller.h"

#include <algorithm>
#include <cmath>

void WheelRestController::begin(const WheelSpeeds& current_speeds) {
    target_speeds_ = current_speeds;
    active_ = true;
}

float WheelRestController::rest_axis(float measured_speed, float& target_speed, float time_step) const {
    const float target_step = tuning_.wheel_rest_deceleration * time_step;
    target_speed = std::copysign(
        std::max(0.0f, std::abs(target_speed) - target_step), target_speed);

    const float desired_acceleration = (target_speed - measured_speed)
                                     / tuning_.wheel_rest_response_time;
    const float limited_acceleration = std::clamp(
        desired_acceleration,
        -tuning_.wheel_rest_deceleration,
        tuning_.wheel_rest_deceleration);
    const float effective_inertia = tuning_.wheel_inertia * tuning_.robot_inertia
                                  / (tuning_.wheel_inertia + tuning_.robot_inertia);
    return effective_inertia * limited_acceleration;
}

WheelTorques WheelRestController::compute_motor_torques(const WheelSpeeds& current_speeds,
                                                         float time_step) {
    if (!active_) {
        return {0.0f, 0.0f, 0.0f};
    }

    return {
        rest_axis(current_speeds.x, target_speeds_.x, time_step),
        rest_axis(current_speeds.y, target_speeds_.y, time_step),
        rest_axis(current_speeds.z, target_speeds_.z, time_step),
    };
}

ReactionWheelController::ReactionWheelController(const ReactionWheelTuning& tuning)
    : tuning_(tuning) {}

TorqueCommand ReactionWheelController::compute_torque(const SensorData& sensor, const WheelSpeeds&,
                                                       float time_step) {
    TorqueCommand cmd;

    // Target is 0.0 for all angles and rates.
    const float roll_error = 0.0f - sensor.roll;
    const float pitch_error = 0.0f - sensor.pitch;
    const float yaw_error = 0.0f - sensor.yaw;

    roll_integral_ = clamp_integral(roll_integral_ + roll_error * time_step);
    pitch_integral_ = clamp_integral(pitch_integral_ + pitch_error * time_step);
    yaw_integral_ = clamp_integral(yaw_integral_ + yaw_error * time_step);

    cmd.tau_x = (tuning_.roll.kp * roll_error)
              + (tuning_.roll.ki * roll_integral_)
              + (tuning_.roll.kd * (0.0f - sensor.roll_rate));
    cmd.tau_y = (tuning_.pitch.kp * pitch_error)
              + (tuning_.pitch.ki * pitch_integral_)
              + (tuning_.pitch.kd * (0.0f - sensor.pitch_rate));
    cmd.tau_z = (tuning_.yaw.kp * yaw_error)
              + (tuning_.yaw.ki * yaw_integral_)
              + (tuning_.yaw.kd * (0.0f - sensor.yaw_rate));

    return cmd;
}

void ReactionWheelController::reset() {
    roll_integral_ = 0.0f;
    pitch_integral_ = 0.0f;
    yaw_integral_ = 0.0f;
}