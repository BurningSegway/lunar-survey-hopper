#include "rw_controller.h"

#include <cmath>
#include <iostream>

int main() {
    const ReactionWheelTuning tuning;
    const LqrWeights weights;
    LqrController controller(tuning, weights);
    const SensorData neutral_sensor{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    const TorqueCommand angle_response = controller.compute_torque(
        {0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, 0.001f);
    const TorqueCommand rate_response = controller.compute_torque(
        {0.0f, 0.0f, 0.0f, 0.1f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, 0.001f);
    const TorqueCommand neutral_response = controller.compute_torque(
        neutral_sensor, {0.0f, 0.0f, 0.0f}, 0.001f);

    if (!(angle_response.tau_x < 0.0f) || !(rate_response.tau_x < 0.0f)
        || neutral_response.tau_x != 0.0f) {
        std::cerr << "LQR should oppose angle and angular-rate errors and return zero at equilibrium"
                  << std::endl;
        return 1;
    }

    if (!std::isfinite(angle_response.tau_x) || !std::isfinite(rate_response.tau_x)) {
        std::cerr << "LQR produced a non-finite torque" << std::endl;
        return 1;
    }

    return 0;
}
