#include "rw_controller.h"

ReactionWheelController::ReactionWheelController(float kp, float kd) 
    : kp_(kp), kd_(kd) {}

TorqueCommand ReactionWheelController::compute_torque(const SensorData& sensor) {
    TorqueCommand cmd;
    
    // Target is 0.0 for all angles and rates
    cmd.tau_x = (kp_ * (0.0f - sensor.roll))  + (kd_ * (0.0f - sensor.roll_rate));
    cmd.tau_y = (kp_ * (0.0f - sensor.pitch)) + (kd_ * (0.0f - sensor.pitch_rate));
    cmd.tau_z = (kp_ * (0.0f - sensor.yaw))   + (kd_ * (0.0f - sensor.yaw_rate));
    
    return cmd;
}