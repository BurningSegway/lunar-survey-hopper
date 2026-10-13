#pragma once

struct SensorData {
    float roll, pitch, yaw;
    float roll_rate, pitch_rate, yaw_rate;
};

struct TorqueCommand {
    float tau_x, tau_y, tau_z;
};

class ReactionWheelController {
private:
    float kp_;
    float kd_;

public:
    ReactionWheelController(float kp, float kd);
    TorqueCommand compute_torque(const SensorData& sensor);
};