#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cmath>
#include <cstring>
#include "rw_controller.h"

// Global pointers and state variables
mjModel* m = nullptr;
mjData* d = nullptr;
ReactionWheelController* controller = nullptr;

int rw_z_motor_id = -1;
int rw_x_motor_id = -1;
int rw_y_motor_id = -1;
int wind_up_motor_id = -1;
int tilt_fl_id = -1;
int tilt_fr_id = -1;

// State machine variables
bool jump_triggered = false;
double jump_start_time = 0.0;
bool tilt_active = false;

// --- SENSOR SIMULATION ---
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

// --- GLFW KEYBOARD INTERRUPT ---
void keyboard_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
        if (!jump_triggered) {
            jump_triggered = true;
            jump_start_time = d->time;
            std::cout << "\n[JUMP INITIATED] Releasing spring!" << std::endl;
        } else {
            std::cout << "[WAIT] The robot is still recovering from the last jump." << std::endl;
        }
    }
    
    if (key == GLFW_KEY_ENTER && action == GLFW_PRESS) {
        tilt_active = !tilt_active;
        std::cout << "\n[TILT] Directional jump tilt is now " << (tilt_active ? "ON (Forward)" : "OFF (Vertical)") << std::endl;
    }
}

// --- CONTROL CALLBACK ---
void control_callback(const mjModel* m, mjData* d) {
    SensorData imu_data = read_simulated_imu(m, d);
    TorqueCommand torques = controller->compute_torque(imu_data);
    
    d->ctrl[rw_x_motor_id] = torques.tau_x;
    d->ctrl[rw_y_motor_id] = torques.tau_y;
    d->ctrl[rw_z_motor_id] = torques.tau_z;

    float leg_force = tilt_active ? 200.0f : 0.0f;
    d->ctrl[tilt_fl_id] = leg_force;
    d->ctrl[tilt_fr_id] = leg_force;

    if (!jump_triggered) {
        d->ctrl[wind_up_motor_id] = -8000;
    } else {
        double time_since_jump = d->time - jump_start_time;
        if (time_since_jump < 0.4) {
            d->ctrl[wind_up_motor_id] = 0;
        } else if (time_since_jump < 3.0) {
            d->ctrl[wind_up_motor_id] = -8000;
        } else {
            jump_triggered = false;
            std::cout << "[READY] Ready for next jump." << std::endl;
        }
    }
}

// --- MAIN SIMULATION LOOP ---
int main() {
    controller = new ReactionWheelController(5.0f, 0.5f);

    char error[1000] = "Could not load XML model";
    m = mj_loadXML("Jumping_Robot.xml", 0, error, 1000);
    if (!m) {
        std::cerr << "\n[FATAL] Failed to load XML: " << error << std::endl;
        return 1;
    }
    d = mj_makeData(m);

    rw_z_motor_id = mj_name2id(m, mjOBJ_ACTUATOR, "motor_rw_z");
    rw_x_motor_id = mj_name2id(m, mjOBJ_ACTUATOR, "motor_rw_x");
    rw_y_motor_id = mj_name2id(m, mjOBJ_ACTUATOR, "motor_rw_y");
    wind_up_motor_id = mj_name2id(m, mjOBJ_ACTUATOR, "wind_up_motor");
    tilt_fl_id = mj_name2id(m, mjOBJ_ACTUATOR, "tilt_fl");
    tilt_fr_id = mj_name2id(m, mjOBJ_ACTUATOR, "tilt_fr");

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

    cam.azimuth = 135;
    cam.elevation = -20;
    cam.distance = 5.0;

    std::cout << "Simulation running." << std::endl;
    std::cout << "Press [ENTER] to toggle forward tilt." << std::endl;
    std::cout << "Press [SPACE] to jump." << std::endl;
    std::cout << "Tip: Press [F2] in the simulation window to toggle MuJoCo's built-in UI overlay." << std::endl;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        mjtNum simstart = d->time;
        while (d->time - simstart < 1.0/60.0) {
            mj_step(m, d);
        }

        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);
        
        glfwSwapBuffers(window);
    }

    mjv_freeScene(&scn);
    mjr_freeContext(&con);
    mj_deleteData(d);
    mj_deleteModel(m);
    glfwTerminate();
    delete controller;
    
    return 0;
}