#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdio>
#include "rw_controller.h"
#include "udp_plotter.h"

// Global pointers and state variables
mjModel* m = nullptr;
mjData* d = nullptr;
ReactionWheelController* controller = nullptr;
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
bool gravity_active = true;
bool control_active = true; // NEW: Flag to enable/disable PD control

// Motor saturation placeholder (rad/s)
const float MAX_WHEEL_VELOCITY = 300.0f; 

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
        gravity_active = !gravity_active;
        m->opt.gravity[2] = gravity_active ? -9.81 : 0.0;
        std::cout << "\n[GRAVITY] " << (gravity_active ? "Normal (-9.81 m/s^2)" : "Zero-G Mode (0.0 m/s^2)") << std::endl;
    }
    
    // NEW: Toggle PD Controller
    if (key == GLFW_KEY_C && action == GLFW_PRESS) {
        control_active = !control_active;
        std::cout << "\n[CONTROLLER] Reaction wheels are now " << (control_active ? "ACTIVE" : "DISABLED") << std::endl;
    }

    if (key == GLFW_KEY_R && action == GLFW_PRESS) {
        mj_resetData(m, d);
        mj_forward(m, d);
        
        jump_triggered = false;
        tilt_active = false;
        jump_start_time = 0.0;
        
        gravity_active = true;
        m->opt.gravity[2] = -9.81;
        
        std::cout << "\n[RESET] Simulation reset to initial state." << std::endl;
    }
}

void control_callback(const mjModel* m, mjData* d) {
    SensorData imu_data = read_simulated_imu(m, d);
    TorqueCommand torques = controller->compute_torque(imu_data);
    
    // Apply torque only if the controller is active
    if (control_active) {
        d->ctrl[rw_x_motor_id] = torques.tau_x;
        d->ctrl[rw_y_motor_id] = torques.tau_y;
        d->ctrl[rw_z_motor_id] = torques.tau_z;
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

int main() {
    controller = new ReactionWheelController(5.0f, 0.5f);
    plotter = new UDPPlotter();

    char error[1000] = "Could not load XML model";
    m = mj_loadXML("Jumping_Robot.xml", 0, error, 1000);
    if (!m) return 1;
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

    std::cout << "Simulation running." << std::endl;
    std::cout << "[SPACE] Jump | [ENTER] Toggle Tilt | [BACKSPACE] Toggle Gravity" << std::endl;
    std::cout << "[C] Toggle Controller | [R] Reset" << std::endl;

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
            "{\"roll\":%.4f, \"roll_ref\":0.0, \"pitch\":%.4f, \"pitch_ref\":0.0, \"yaw\":%.4f, \"yaw_ref\":0.0, "
            "\"vel_x\":%.4f, \"vel_y\":%.4f, \"vel_z\":%.4f, \"sat_max\":%.1f, \"sat_min\":%.1f}",
            imu.roll, imu.pitch, imu.yaw, 
            vel_x, vel_y, vel_z, MAX_WHEEL_VELOCITY, -MAX_WHEEL_VELOCITY);
        
        plotter->send(std::string(json_buffer));

        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);
        glfwSwapBuffers(window);
    }

    mjv_freeScene(&scn); mjr_freeContext(&con);
    mj_deleteData(d); mj_deleteModel(m);
    glfwTerminate();
    delete controller; delete plotter;
    return 0;
}