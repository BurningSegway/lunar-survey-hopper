#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include "rw_controller.h"
#include "udp_plotter.h"

// Global pointers and state variables
mjModel* m = nullptr;
mjData* d = nullptr;
std::unique_ptr<AttitudeController> controller;
UDPPlotter* plotter = nullptr;

// Actuator IDs
int rw_z_motor_id = -1, rw_x_motor_id = -1, rw_y_motor_id = -1;
int wind_up_motor_id = -1, tilt_fl_id = -1, tilt_fr_id = -1;

// Joint IDs for reading wheel velocities
int rw_x_joint_id = -1, rw_y_joint_id = -1, rw_z_joint_id = -1;

// State machine variables
bool jump_triggered = false;
double jump_start_time = 0.0;
bool tilt_active = false;
constexpr mjtNum earth_gravity = 9.81;
constexpr mjtNum moon_gravity = 1.62;
mjtNum selected_gravity = earth_gravity;
mjtNum launch_gravity = earth_gravity;
enum class WheelControlMode { Stabilizing, Resting, Disabled };
WheelControlMode wheel_control_mode = WheelControlMode::Stabilizing;
float yaw_reference = 0.0f;

ReactionWheelTuning tuning_config;
std::unique_ptr<WheelRestController> wheel_rest_controller;

float wheel_effective_inertia() {
    return tuning_config.wheel_inertia * tuning_config.robot_inertia
         / (tuning_config.wheel_inertia + tuning_config.robot_inertia);
}

float limit_wheel_command(const mjModel* model, const mjData* data, int joint_id, float command) {
    const int dof_id = model->jnt_dofadr[joint_id];
    const float wheel_velocity = data->qvel[dof_id];
    const float time_step = static_cast<float>(model->opt.timestep);
    const float effective_inertia = wheel_effective_inertia();
    // The motor torque accelerates the wheel relative to the robot body.
    const float max_motor_torque = effective_inertia * tuning_config.max_wheel_acceleration;
    const float limited_command = std::clamp(command, -max_motor_torque, max_motor_torque);

    if (wheel_velocity * limited_command <= 0.0f) {
        return limited_command;
    }

    const float speed_headroom = tuning_config.max_wheel_velocity - std::abs(wheel_velocity);
    if (speed_headroom <= 0.0f) {
        return 0.0f;
    }

    const float allowed_acceleration = speed_headroom / time_step;
    const float allowed_torque = effective_inertia
                               * std::min(tuning_config.max_wheel_acceleration, allowed_acceleration);
    return std::clamp(limited_command, -allowed_torque, allowed_torque);
}

float combine_attitude_and_rest_torque(float attitude_torque, float rest_torque) {
    const float max_motor_torque = wheel_effective_inertia() * tuning_config.max_wheel_acceleration;
    const float available_rest_torque = std::clamp(
        rest_torque, -max_motor_torque - attitude_torque, max_motor_torque - attitude_torque);
    return attitude_torque + available_rest_torque;
}

SensorData read_simulated_imu(const mjModel* m, const mjData* d) {
    SensorData sensor;
    float qw = d->qpos[3], qx = d->qpos[4], qy = d->qpos[5], qz = d->qpos[6];
    
    sensor.roll = std::atan2(2.0f * (qw * qx + qy * qz), 1.0f - 2.0f * (qx * qx + qy * qy));
    float sinp = 2.0f * (qw * qy - qz * qx);
    sensor.pitch = std::abs(sinp) >= 1.0f ? std::copysign(1.57079632679f, sinp) : std::asin(sinp);
    sensor.yaw = std::atan2(2.0f * (qw * qz + qx * qy), 1.0f - 2.0f * (qy * qy + qz * qz));

    sensor.roll_rate  = d->qvel[3];
    sensor.pitch_rate = d->qvel[4];
    sensor.yaw_rate   = d->qvel[5]; 
    return sensor;
}

float wrapped_yaw_error(float yaw, float reference) {
    constexpr float two_pi = 6.28318530718f;
    return std::remainder(yaw - reference, two_pi);
}

WheelSpeeds read_wheel_speeds(const mjModel* model, const mjData* data) {
    return {
        static_cast<float>(data->qvel[model->jnt_dofadr[rw_x_joint_id]]),
        static_cast<float>(data->qvel[model->jnt_dofadr[rw_y_joint_id]]),
        static_cast<float>(data->qvel[model->jnt_dofadr[rw_z_joint_id]]),
    };
}

void keyboard_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
        if (!jump_triggered) {
            jump_triggered = true;
            jump_start_time = d->time;
            std::cout << "\n[JUMP INITIATED] Releasing spring!" << std::endl;
        }
    }
    
    if (key == GLFW_KEY_ENTER && action == GLFW_PRESS) {
        tilt_active = !tilt_active;
        std::cout << "\n[TILT] Directional jump tilt is now " << (tilt_active ? "ON (Forward)" : "OFF (Vertical)") << std::endl;
    }
    
    if (key == GLFW_KEY_BACKSPACE && action == GLFW_PRESS) {
        selected_gravity = selected_gravity == earth_gravity ? moon_gravity : earth_gravity;
        m->opt.gravity[2] = -selected_gravity;
        std::cout << "\n[GRAVITY] Switched to "
                  << (selected_gravity == moon_gravity ? "Moon" : "Earth")
                  << " (" << selected_gravity << " m/s^2)." << std::endl;
    }
    
    if (key == GLFW_KEY_C && action == GLFW_PRESS) {
        if (wheel_control_mode == WheelControlMode::Stabilizing) {
            yaw_reference = read_simulated_imu(m, d).yaw;
            wheel_rest_controller->begin(read_wheel_speeds(m, d));
            wheel_control_mode = WheelControlMode::Resting;
            std::cout << "\n[WHEELS] Controlled rest requested; holding current yaw at "
                      << yaw_reference << " rad while attitude stabilization remains active." << std::endl;
        } else if (wheel_control_mode == WheelControlMode::Resting) {
            wheel_rest_controller->cancel();
            wheel_control_mode = WheelControlMode::Stabilizing;
            std::cout << "\n[WHEELS] Rest request cancelled; attitude stabilization remains active." << std::endl;
        } else {
            wheel_control_mode = WheelControlMode::Stabilizing;
            std::cout << "\n[CONTROLLER] Attitude stabilization enabled." << std::endl;
        }
    }

    if (key == GLFW_KEY_X && action == GLFW_PRESS) {
        wheel_rest_controller->cancel();
        wheel_control_mode = WheelControlMode::Disabled;
        controller->reset();
        std::cout << "\n[CONTROLLER] Disabled; reaction-wheel motor torques are zero." << std::endl;
    }

    if (key == GLFW_KEY_R && action == GLFW_PRESS) {
        mj_resetData(m, d);
        mj_forward(m, d);
        controller->reset();
        wheel_rest_controller->cancel();
        wheel_control_mode = WheelControlMode::Stabilizing;
        yaw_reference = 0.0f;
        
        jump_triggered = false;
        tilt_active = false;
        jump_start_time = 0.0;
        
        selected_gravity = launch_gravity;
        m->opt.gravity[2] = -launch_gravity;
        
        std::cout << "\n[RESET] Simulation reset to initial state." << std::endl;
    }
}

void control_callback(const mjModel* m, mjData* d) {
    if (wheel_control_mode != WheelControlMode::Disabled) {
        SensorData imu_data = read_simulated_imu(m, d);
        imu_data.yaw = wrapped_yaw_error(imu_data.yaw, yaw_reference);
        const WheelSpeeds wheel_speeds = read_wheel_speeds(m, d);
        const TorqueCommand torques = controller->compute_torque(
            imu_data, wheel_speeds, static_cast<float>(m->opt.timestep));
        const WheelTorques rest_torques = wheel_rest_controller->compute_motor_torques(
            wheel_speeds, static_cast<float>(m->opt.timestep));

        d->ctrl[rw_x_motor_id] = limit_wheel_command(
            m, d, rw_x_joint_id, combine_attitude_and_rest_torque(-torques.tau_x, rest_torques.x));
        d->ctrl[rw_y_motor_id] = limit_wheel_command(
            m, d, rw_y_joint_id, combine_attitude_and_rest_torque(-torques.tau_y, rest_torques.y));
        d->ctrl[rw_z_motor_id] = limit_wheel_command(
            m, d, rw_z_joint_id, combine_attitude_and_rest_torque(-torques.tau_z, rest_torques.z));
    } else {
        d->ctrl[rw_x_motor_id] = 0.0f;
        d->ctrl[rw_y_motor_id] = 0.0f;
        d->ctrl[rw_z_motor_id] = 0.0f;
    }

    d->ctrl[tilt_fl_id] = tilt_active ? 200.0f : 0.0f;
    d->ctrl[tilt_fr_id] = tilt_active ? 200.0f : 0.0f;

    if (!jump_triggered) {
        d->ctrl[wind_up_motor_id] = -8000;
    } else {
        double time_since_jump = d->time - jump_start_time;
        if (time_since_jump < 0.4) d->ctrl[wind_up_motor_id] = 0;
        else if (time_since_jump < 3.0) d->ctrl[wind_up_motor_id] = -8000;
        else jump_triggered = false;
    }
}

int main(int argc, char* argv[]) {
    std::string controller_name = "pid";
    std::string gravity_name = "earth";
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help") {
            std::cout << "Usage: jumper_sim [--controller pid|lqr] [--gravity earth|moon]" << std::endl;
            return 0;
        }
        if (argument == "--controller" && i + 1 < argc) {
            controller_name = argv[++i];
        } else if (argument.rfind("--controller=", 0) == 0) {
            controller_name = argument.substr(std::string("--controller=").size());
        } else if (argument == "--gravity" && i + 1 < argc) {
            gravity_name = argv[++i];
        } else if (argument.rfind("--gravity=", 0) == 0) {
            gravity_name = argument.substr(std::string("--gravity=").size());
        } else {
            std::cerr << "[ERROR] Unknown or incomplete argument: " << argument
                      << "\nUsage: jumper_sim [--controller pid|lqr] [--gravity earth|moon]" << std::endl;
            return 2;
        }
    }
    if (controller_name != "pid" && controller_name != "lqr") {
        std::cerr << "[ERROR] Unknown controller '" << controller_name
                  << "'. Choose 'pid' or 'lqr'." << std::endl;
        return 2;
    }
    if (gravity_name != "earth" && gravity_name != "moon") {
        std::cerr << "[ERROR] Unknown gravity '" << gravity_name
                  << "'. Choose 'earth' or 'moon'." << std::endl;
        return 2;
    }
    selected_gravity = gravity_name == "moon" ? moon_gravity : earth_gravity;
    launch_gravity = selected_gravity;

    wheel_rest_controller = std::make_unique<WheelRestController>(tuning_config);

    plotter = new UDPPlotter();

    std::cout << "[TUNING] max_wheel_velocity=" << tuning_config.max_wheel_velocity
              << " rad/s (" << tuning_config.max_wheel_velocity / ReactionWheelTuning::radians_per_turn << " turns/s)"
              << " max_wheel_acceleration=" << tuning_config.max_wheel_acceleration
              << " rad/s^2 (" << tuning_config.max_wheel_acceleration / ReactionWheelTuning::radians_per_turn << " turns/s^2)"
              << " wheel_rest_deceleration=" << tuning_config.wheel_rest_deceleration
              << " rad/s^2 (" << tuning_config.wheel_rest_deceleration / ReactionWheelTuning::radians_per_turn << " turns/s^2)"
              << " wheel_rest_response_time=" << tuning_config.wheel_rest_response_time << " s"
              << " wheel_torque_limit=" << wheel_effective_inertia() * tuning_config.max_wheel_acceleration << " Nm"
              << " robot_mass=" << tuning_config.robot_mass
              << " robot_inertia=" << tuning_config.robot_inertia
              << " wheel_mass=" << tuning_config.wheel_mass
              << " wheel_inertia=" << tuning_config.wheel_inertia << std::endl;

    char error[1000] = "Could not load XML model";
    std::filesystem::path model_path = std::filesystem::path("Jumping_Robot.xml");
    if (!std::filesystem::exists(model_path)) {
        model_path = std::filesystem::path("build") / "Jumping_Robot.xml";
    }
    if (!std::filesystem::exists(model_path)) {
        std::cerr << "[ERROR] Could not find Jumping_Robot.xml in the project or build directory." << std::endl;
        return 1;
    }

    m = mj_loadXML(model_path.string().c_str(), 0, error, 1000);
    if (!m) {
        std::cerr << "[ERROR] Failed to load model: " << error << std::endl;
        return 1;
    }
    m->opt.gravity[2] = -selected_gravity;

    d = mj_makeData(m);

    rw_z_motor_id = mj_name2id(m, mjOBJ_ACTUATOR, "motor_rw_z");
    rw_x_motor_id = mj_name2id(m, mjOBJ_ACTUATOR, "motor_rw_x");
    rw_y_motor_id = mj_name2id(m, mjOBJ_ACTUATOR, "motor_rw_y");
    wind_up_motor_id = mj_name2id(m, mjOBJ_ACTUATOR, "wind_up_motor");
    tilt_fl_id = mj_name2id(m, mjOBJ_ACTUATOR, "tilt_fl");
    tilt_fr_id = mj_name2id(m, mjOBJ_ACTUATOR, "tilt_fr");

    rw_x_joint_id = mj_name2id(m, mjOBJ_JOINT, "rw_x_joint");
    rw_y_joint_id = mj_name2id(m, mjOBJ_JOINT, "rw_y_joint");
    rw_z_joint_id = mj_name2id(m, mjOBJ_JOINT, "rw_z_joint");

    const LqrWeights lqr_weights;
    try {
        if (controller_name == "lqr") {
            auto lqr_controller = std::make_unique<LqrController>(tuning_config, lqr_weights);
            const auto& gains = lqr_controller->gains();
            std::cout << "[LQR] weights=(angle=" << lqr_weights.angle
                      << ", angular_rate=" << lqr_weights.angular_rate
                      << ", torque=" << lqr_weights.torque << ") gains=("
                      << gains[0] << ", " << gains[1] << ")"
                      << std::endl;
            controller = std::move(lqr_controller);
        } else {
            std::cout << "[PID] roll=(" << tuning_config.roll.kp << ", " << tuning_config.roll.ki
                      << ", " << tuning_config.roll.kd << ") "
                      << "pitch=(" << tuning_config.pitch.kp << ", " << tuning_config.pitch.ki
                      << ", " << tuning_config.pitch.kd << ") "
                      << "yaw=(" << tuning_config.yaw.kp << ", " << tuning_config.yaw.ki
                      << ", " << tuning_config.yaw.kd << ")" << std::endl;
            controller = std::make_unique<ReactionWheelController>(tuning_config);
        }
    } catch (const std::exception& exception) {
        std::cerr << "[ERROR] Could not initialize " << controller_name
                  << " controller: " << exception.what() << std::endl;
        mj_deleteData(d);
        mj_deleteModel(m);
        return 1;
    }
    std::cout << "[CONTROLLER] Selected: " << controller_name << std::endl;
    std::cout << "[GRAVITY] Selected: " << gravity_name << " (" << selected_gravity << " m/s^2)" << std::endl;

    mjcb_control = control_callback;

    if (!glfwInit()) return 1;
    GLFWwindow* window = glfwCreateWindow(1200, 900, "Sim2Real - Directional Jumper", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); 
    glfwSetKeyCallback(window, keyboard_callback);

    mjvScene scn; mjv_defaultScene(&scn);
    mjvCamera cam; mjv_defaultCamera(&cam);
    mjvOption opt; mjv_defaultOption(&opt);
    mjrContext con; mjr_defaultContext(&con);
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    cam.azimuth = 135; cam.elevation = -20; cam.distance = 5.0;
    const int base_body_id = mj_name2id(m, mjOBJ_BODY, "base_link");

    std::cout << "Simulation running." << std::endl;
    std::cout << "[SPACE] Jump | [ENTER] Toggle Tilt | [BACKSPACE] Toggle Earth/Moon Gravity" << std::endl;
    std::cout << "[C] Request/cancel controlled wheel rest | [X] Disable/zero motor torque | [R] Reset" << std::endl;

    char json_buffer[512];

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        mjtNum simstart = d->time;
        while (d->time - simstart < 1.0/60.0) mj_step(m, d);

        float vel_x = d->qvel[m->jnt_dofadr[rw_x_joint_id]];
        float vel_y = d->qvel[m->jnt_dofadr[rw_y_joint_id]];
        float vel_z = d->qvel[m->jnt_dofadr[rw_z_joint_id]];
        SensorData imu = read_simulated_imu(m, d);

        snprintf(json_buffer, sizeof(json_buffer), 
            "{\"roll\":%.4f, \"roll_ref\":0.0, \"pitch\":%.4f, \"pitch_ref\":0.0, \"yaw\":%.4f, \"yaw_ref\":%.4f, "
            "\"vel_x\":%.4f, \"vel_y\":%.4f, \"vel_z\":%.4f, \"sat_max\":%.1f, \"sat_min\":%.1f}",
            imu.roll, imu.pitch, imu.yaw, yaw_reference,
            vel_x, vel_y, vel_z, tuning_config.max_wheel_velocity, -tuning_config.max_wheel_velocity);
        
        plotter->send(std::string(json_buffer));

        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
        for (int axis = 0; axis < 3; ++axis) {
            cam.lookat[axis] = d->xpos[3 * base_body_id + axis];
        }
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);
        glfwSwapBuffers(window);
    }

    mjv_freeScene(&scn); mjr_freeContext(&con);
    mj_deleteData(d); mj_deleteModel(m);
    glfwTerminate();
    controller.reset();
    delete plotter;
    return 0;
}