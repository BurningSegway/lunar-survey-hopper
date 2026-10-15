#pragma once

#include <array>

struct PidGains {
    float kp = 1.1f;
    float ki = 0.15f;
    float kd = 0.13f;
};

struct ReactionWheelTuning {
    static constexpr float radians_per_turn = 6.28318530718f;
    // Teensy PID output limit (-10 to 10), interpreted as ODrive turns/s.
    float max_wheel_velocity = 20.0f * radians_per_turn;
    // ODrive velocity ramp rate, modeled as a torque/acceleration cap in this torque-driven sim.
    float max_wheel_acceleration = 75.0f * radians_per_turn;
    // Lower acceleration target used only while requesting controlled wheel rest.
    float wheel_rest_deceleration = 10.0f * radians_per_turn;
    float wheel_rest_response_time = 0.25f;

    // Keep these aligned with the MuJoCo XML inertials in Jumping_Robot.xml.
    float robot_mass = 5.0f;
    float robot_inertia = 0.12f;
    float wheel_mass = 0.2f;
    float wheel_inertia = 0.0036f;

    PidGains roll = {1.1f, 0.15f, 0.13f};
    PidGains pitch = {1.1f, 0.15f, 0.13f};
    PidGains yaw = {1.1f, 0.15f, 0.13f};
};

struct LqrWeights {
    float angle = 100.0f;
    float angular_rate = 10.0f;
    float torque = 1.0f;
};

struct SensorData {
    float roll, pitch, yaw;
    float roll_rate, pitch_rate, yaw_rate;
};

struct TorqueCommand {
    float tau_x, tau_y, tau_z;
};

struct WheelSpeeds {
    float x, y, z;
};

struct WheelTorques {
    float x, y, z;
};

class WheelRestController {
private:
    ReactionWheelTuning tuning_;
    WheelSpeeds target_speeds_{0.0f, 0.0f, 0.0f};
    bool active_ = false;

    float rest_axis(float measured_speed, float& target_speed, float time_step) const;

public:
    explicit WheelRestController(const ReactionWheelTuning& tuning) : tuning_(tuning) {}

    void begin(const WheelSpeeds& current_speeds);
    void cancel() { active_ = false; }
    WheelSpeeds target_speeds() const { return target_speeds_; }
    WheelTorques compute_motor_torques(const WheelSpeeds& current_speeds, float time_step);
};

class AttitudeController {
public:
    virtual ~AttitudeController() = default;
    // Controller outputs are desired torques on the robot body, not the wheel actuators.
    virtual TorqueCommand compute_torque(const SensorData& sensor, const WheelSpeeds& wheel_speeds,
                                         float time_step) = 0;
    virtual void reset() = 0;
};

class ReactionWheelController final : public AttitudeController {
private:
    ReactionWheelTuning tuning_;
    float roll_integral_ = 0.0f;
    float pitch_integral_ = 0.0f;
    float yaw_integral_ = 0.0f;
    static constexpr float kMaxIntegral = 10.0f;

    static float clamp_integral(float value) {
        if (value > kMaxIntegral) return kMaxIntegral;
        if (value < -kMaxIntegral) return -kMaxIntegral;
        return value;
    }

public:
    explicit ReactionWheelController(const ReactionWheelTuning& tuning = {});
    const ReactionWheelTuning& tuning() const { return tuning_; }
    TorqueCommand compute_torque(const SensorData& sensor, const WheelSpeeds& wheel_speeds,
                                 float time_step) override;
    void reset() override;
};

class LqrController final : public AttitudeController {
private:
    ReactionWheelTuning tuning_;
    LqrWeights weights_;
    std::array<double, 2> gains_{};

    std::array<double, 2> compute_axis_gains() const;

public:
    LqrController(const ReactionWheelTuning& tuning, const LqrWeights& weights);
    TorqueCommand compute_torque(const SensorData& sensor, const WheelSpeeds& wheel_speeds,
                                 float time_step) override;
    void reset() override {}

    const std::array<double, 2>& gains() const { return gains_; }
    const LqrWeights& weights() const { return weights_; }
};