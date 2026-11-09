
#include <Arduino.h>
#include <math.h>
#include "ODriveCAN.h"

//MTI includes
#include "xsens_mti.h"      // Main library
#include "xsens_utility.h"  // Needed for quaternion conversion function

#include "telemetry.h"

/* Telemetry channels ----------------------------------------------------------
 * Switchable at runtime from the web dashboard (or a serial monitor, see
 * telemetry.h). The 'enabled' column is the default after boot.
 * To add a print: add an enum entry + table row (same order), then print it in
 * the telemetry block at the end of loop().
 */
enum TelemetryChannelId : size_t {
  TLM_IMU_AGE,
  TLM_LQR_TORQUE,
  TLM_LQR_STATE,
  TLM_FLIGHT,
  TLM_EULER,
  TLM_QUAT,
  TLM_ACCEL,
  TLM_BUS_VOLTAGE,
  TLM_WHEELS,
  TLM_BURN,
  TLM_BALLISTIC,
  TLM_COUNT
};

TelemetryChannel telemetryChannels[] = {
  // name           variables                                                   enabled
  { "imu_age",      "QuatAgeMs GyroAgeMs",                                      true  },
  { "lqr_torque",   "LQR-tau-x LQR-tau-y LQR-tau-z",                            true  },
  { "lqr_state",    "qe-x qe-y qe-z omega-x omega-y omega-z",                   false },
  { "flight",       "AccelerationMagnitude Airborne",                           true  },
  { "euler",        "Roll Pitch Yaw",                                           false },
  { "quat",         "q-w q-x q-y q-z",                                          false },
  { "accel",        "AccelX AccelY AccelZ",                                     false },
  { "bus_voltage",  "BusVoltage",                                               false },
  { "wheels",       "odrv0-pos odrv0-vel odrv1-pos odrv1-vel odrv2-pos odrv2-vel", false },
  { "burn",         "AccLinE AccLinN AccLinU VelE VelN VelU PosE PosN PosU",    false },
  { "ballistic",    "V0E V0N V0U V0Speed V0Elevation V0Heading ApexTime ApexHeight FlightTime Range LandE LandN PredE PredN PredU", false },
};
static_assert(sizeof(telemetryChannels) / sizeof(telemetryChannels[0]) == TLM_COUNT,
              "telemetryChannels must have one row per TelemetryChannelId");

// 20 ms = 50 Hz default print rate (changeable at runtime with "tlm rate <ms>")
Telemetry telemetry(telemetryChannels, TLM_COUNT, 20);

//MTI vars
// Cache a copy of IMU data
float    temperature     = 0;       // in degress celcius
uint32_t pressure        = 0;       // in pascals
float    euler_pry[3]    = { 0 };   // -180 to +180 degress
float    acceleration[3] = { 0 };   // in m/s^2
uint32_t lastAccelerationMs = 0;
uint32_t lastAccelerationUs = 0;    // sample timestamp for the velocity integration (dt)
uint32_t accelerationSampleCount = 0;

// Attitude quaternion as delivered by the MTi: scalar-first {w, x, y, z}.
float    quaternion_wxyz[4] = { 1.0f, 0.0f, 0.0f, 0.0f };
uint32_t lastQuaternionMs   = 0;
bool     receivedQuaternion = false;

// Body angular rate (rate of turn) in rad/s, sensor frame.
// NOTE: the MTi must be configured to output "Rate of Turn" for this to update.
float    gyro_rad_s[3]   = { 0 };
uint32_t lastGyroMs      = 0;
bool     receivedGyro    = false;

constexpr float GRAVITY_M_S2 = 9.81f;
constexpr float LAUNCH_ACCEL_THRESHOLD_M_S2 = 1.25f * GRAVITY_M_S2;
constexpr uint32_t LAUNCH_DEBOUNCE_MS = 150;
constexpr float PROPULSION_END_ACCEL_THRESHOLD_M_S2 = 1.1f * GRAVITY_M_S2; //tjek når accleration falder, før hjul tænder
constexpr float FLIGHT_ACCEL_THRESHOLD_M_S2 = 0.75f * GRAVITY_M_S2;
constexpr uint32_t FLIGHT_DEBOUNCE_MS = 50;
constexpr float LANDING_DECEL_THRESHOLD_M_S2 = GRAVITY_M_S2 / 1.25f;
constexpr float LANDING_ACCEL_THRESHOLD_M_S2 = 1.75f * GRAVITY_M_S2;
constexpr uint32_t LANDING_DEBOUNCE_MS = 30;
constexpr uint32_t ACCELERATION_STALE_TIMEOUT_MS = 100;

enum class FlightState {
  WAITING_FOR_LAUNCH,
  WAITING_FOR_PROPULSION_CUTOFF,
  AIRBORNE_WAITING_FOR_FLIGHT,
  AIRBORNE_LANDING_ARMED
};

FlightState flightState = FlightState::WAITING_FOR_LAUNCH;
float accelerationMagnitude = 0.0f;

void setAttitudeControlEnabled(bool enabled);

/* Trajectory estimate (Euler integration of acceleration) ---------------------
 * The MTi "Acceleration" output is the bias-compensated specific force in the
 * sensor frame (reads ~+g on the up axis at rest). Every sample is rotated into
 * the local ENU frame with the MTi attitude quaternion (sensor -> local) and
 * gravity is removed from the up axis, giving the linear acceleration a.
 *
 *   Forward Euler:  p[k] = p[k-1] + v[k-1] * dt[k]
 *                   v[k] = v[k-1] + a[k-1] * dt[k]
 *
 * Integration only runs during the burn: origin p = 0 is the launch point,
 * it starts on the first above-threshold launch sample (including the step
 * from the previous sample, so the launch debounce loses nothing), and a
 * rejected candidate resets to rest.
 *
 * At propulsion cutoff p and v are latched (p_c, v0) and integration stops:
 * after cutoff only gravity acts (drag neglected), so the flight is fully
 * given by the drag-free ballistic arc
 *   p_pred(t) = p_c + v0 * t - [0, 0, g/2] * t^2,   t = time since cutoff
 * which is evaluated every IMU sample until landing (streamed on the
 * 'ballistic' channel; t is capped at the predicted flight time, so it stops
 * at the predicted landing point) and held afterwards until the next
 * confirmed launch.
 * v0 is also reported as speed, elevation above horizontal and compass
 * heading (0 = north, 90 = east).
 */
constexpr float BALLISTIC_GRAVITY_M_S2 = GRAVITY_M_S2;  // g for the trajectory prediction
constexpr float VELOCITY_MAX_DT_S      = 0.05f;          // larger sample gaps are not integrated

float accel_lin_world[3]   = { 0 };   // linear acceleration, ENU [m/s^2]
float velocity_world[3]    = { 0 };   // ENU [m/s]
float position_world[3]    = { 0 };   // ENU [m], relative to the launch point
bool  trajectoryIntegrating = false;

float    prevAccelLinWorld[3]  = { 0 };
uint32_t prevVelocitySampleUs  = 0;
bool     hasPrevVelocitySample = false;

// Latched at propulsion cutoff
float    launch_velocity[3]   = { 0 };   // v0, ENU [m/s]
float    launchSpeed          = 0.0f;    // |v0| [m/s]
float    launchElevationDeg   = 0.0f;    // angle of v0 above horizontal [deg]
float    launchHeadingDeg     = 0.0f;    // compass heading of v0, 0 = N, 90 = E [deg]
float    cutoff_position[3]   = { 0 };   // p_c, ENU [m]
uint32_t cutoffSampleUs       = 0;
uint32_t cutoffSampleMs       = 0;
bool     launchVelocityValid  = false;
bool     ballisticActive      = false;   // cutoff -> landing: prediction is updated
float    predicted_position[3] = { 0 };  // p_pred(t), ENU [m]
float    ballisticApexTimeS    = 0.0f;   // time from cutoff to apex
float    ballisticApexHeightM  = 0.0f;   // apex height above the launch point
float    ballisticFlightTimeS  = 0.0f;   // time from cutoff back to launch height
float    ballisticRangeM       = 0.0f;   // horizontal distance from launch point at that time
float    ballisticLanding[2]   = { 0 };  // predicted landing point E, N [m]

// Rotate a sensor-frame vector into the local (ENU) frame: out = R(q) * v,
// q scalar-first {w, x, y, z}, sensor -> local (MTi convention).
void rotateSensorToWorld(const float q[4], const float v[3], float out[3]) {
  const float w = q[0], x = q[1], y = q[2], z = q[3];
  out[0] = (1.0f - 2.0f * (y * y + z * z)) * v[0] + 2.0f * (x * y - w * z) * v[1] + 2.0f * (x * z + w * y) * v[2];
  out[1] = 2.0f * (x * y + w * z) * v[0] + (1.0f - 2.0f * (x * x + z * z)) * v[1] + 2.0f * (y * z - w * x) * v[2];
  out[2] = 2.0f * (x * z - w * y) * v[0] + 2.0f * (y * z + w * x) * v[1] + (1.0f - 2.0f * (x * x + y * y)) * v[2];
}

void resetTrajectoryEstimate() {
  trajectoryIntegrating = false;
  ballisticActive = false;
  hasPrevVelocitySample = false;
  for (size_t i = 0; i < 3; ++i) {
    velocity_world[i] = 0.0f;
    position_world[i] = 0.0f;
    accel_lin_world[i] = 0.0f;
    prevAccelLinWorld[i] = 0.0f;
  }
}

// Rotates the newest sample to ENU, removes gravity and returns the time step
// since the previous sample (0 if there is no usable previous sample).
float measureLinearAcceleration(uint32_t sampleUs) {
  rotateSensorToWorld(quaternion_wxyz, acceleration, accel_lin_world);
  accel_lin_world[2] -= GRAVITY_M_S2;

  float dt = 0.0f;
  if (hasPrevVelocitySample) {
    dt = (uint32_t)(sampleUs - prevVelocitySampleUs) * 1e-6f;
    if (dt > VELOCITY_MAX_DT_S) dt = 0.0f;
  }
  return dt;
}

// One forward-Euler step using the previous sample's velocity / acceleration.
void eulerStepTrajectory(float dt) {
  for (size_t i = 0; i < 3; ++i) {
    position_world[i] += velocity_world[i] * dt;
    velocity_world[i] += prevAccelLinWorld[i] * dt;
  }
}

// Launch candidate: start from rest at the origin. The previous hop's
// ballistic result (launchVelocityValid) stays valid until the launch is
// confirmed, so a touchdown shock that only looks like a launch for a few
// samples does not wipe it.
void beginTrajectory(float dt) {
  trajectoryIntegrating = true;
  ballisticActive = false;
  for (size_t i = 0; i < 3; ++i) {
    velocity_world[i] = 0.0f;
    position_world[i] = 0.0f;
  }
  eulerStepTrajectory(dt);
}

// Called at the end of every processed sample.
void finishTrajectorySample(uint32_t sampleUs) {
  for (size_t i = 0; i < 3; ++i) {
    prevAccelLinWorld[i] = accel_lin_world[i];
  }
  prevVelocitySampleUs = sampleUs;
  hasPrevVelocitySample = true;
}

void updatePredictedPosition(uint32_t sampleUs) {
  float t = (uint32_t)(sampleUs - cutoffSampleUs) * 1e-6f;
  // Stop at the predicted landing point instead of continuing below ground
  // when the real touchdown is detected later than predicted.
  if (ballisticFlightTimeS > 0.0f && t > ballisticFlightTimeS) t = ballisticFlightTimeS;
  for (size_t i = 0; i < 3; ++i) {
    predicted_position[i] = cutoff_position[i] + launch_velocity[i] * t;
  }
  predicted_position[2] -= 0.5f * BALLISTIC_GRAVITY_M_S2 * t * t;
}

// Propulsion cutoff: latch p_c / v0, stop integrating and predict the
// drag-free ballistic arc.
void latchLaunchVelocityAndPredict(uint32_t sampleUs, uint32_t sampleMs) {
  for (size_t i = 0; i < 3; ++i) {
    launch_velocity[i] = velocity_world[i];
    cutoff_position[i] = position_world[i];
  }
  cutoffSampleUs = sampleUs;
  cutoffSampleMs = sampleMs;
  trajectoryIntegrating = false;
  launchVelocityValid = true;
  ballisticActive = true;
  updatePredictedPosition(sampleUs);

  const float g = BALLISTIC_GRAVITY_M_S2;
  const float vUp = launch_velocity[2];
  const float h0 = cutoff_position[2];
  const float vHorizontal = sqrtf(launch_velocity[0] * launch_velocity[0] +
                                  launch_velocity[1] * launch_velocity[1]);

  launchSpeed        = sqrtf(vHorizontal * vHorizontal + vUp * vUp);
  launchElevationDeg = atan2f(vUp, vHorizontal) * (180.0f / PI);
  launchHeadingDeg   = atan2f(launch_velocity[0], launch_velocity[1]) * (180.0f / PI);
  if (launchHeadingDeg < 0.0f) launchHeadingDeg += 360.0f;

  ballisticApexTimeS   = (vUp > 0.0f) ? vUp / g : 0.0f;
  ballisticApexHeightM = h0 + ((vUp > 0.0f) ? vUp * vUp / (2.0f * g) : 0.0f);
  // h0 + vUp*t - g/2 t^2 = 0  ->  positive root
  const float disc = vUp * vUp + 2.0f * g * h0;
  ballisticFlightTimeS = (disc > 0.0f) ? (vUp + sqrtf(disc)) / g : 0.0f;
  if (ballisticFlightTimeS < 0.0f) ballisticFlightTimeS = 0.0f;
  ballisticLanding[0] = cutoff_position[0] + launch_velocity[0] * ballisticFlightTimeS;
  ballisticLanding[1] = cutoff_position[1] + launch_velocity[1] * ballisticFlightTimeS;
  ballisticRangeM = sqrtf(ballisticLanding[0] * ballisticLanding[0] +
                          ballisticLanding[1] * ballisticLanding[1]);

  
  Serial.print("Cutoff: v0 ENU [m/s] ");
  Serial.print(launch_velocity[0], 3); Serial.print(' ');
  Serial.print(launch_velocity[1], 3); Serial.print(' ');
  Serial.print(launch_velocity[2], 3);
  Serial.print(" (");
  Serial.print(launchSpeed, 3); Serial.print(" m/s, elev ");
  Serial.print(launchElevationDeg, 1); Serial.print(" deg, hdg ");
  Serial.print(launchHeadingDeg, 1); Serial.print(" deg)");
  Serial.print(", p ENU [m] ");
  Serial.print(cutoff_position[0], 3); Serial.print(' ');
  Serial.print(cutoff_position[1], 3); Serial.print(' ');
  Serial.print(cutoff_position[2], 3);
  Serial.print(" | predicted apex ");
  Serial.print(ballisticApexHeightM, 3); Serial.print(" m @ ");
  Serial.print(ballisticApexTimeS, 3); Serial.print(" s, flight ");
  Serial.print(ballisticFlightTimeS, 3); Serial.print(" s, range ");
  Serial.print(ballisticRangeM, 3); Serial.print(" m, landing EN [m] ");
  Serial.print(ballisticLanding[0], 3); Serial.print(' ');
  Serial.print(ballisticLanding[1], 3); Serial.println();
}

// Landing: freeze the prediction and compare the measured flight time
// (cutoff -> first touchdown sample) with the predicted one.
void endTrajectory(uint32_t touchdownMs) {
  trajectoryIntegrating = false;
  if (!ballisticActive) return;
  ballisticActive = false;

  Serial.print("Landing: flight time measured ");
  Serial.print((uint32_t)(touchdownMs - cutoffSampleMs) * 1e-3f, 3);
  Serial.print(" s, predicted ");
  Serial.print(ballisticFlightTimeS, 3);
  Serial.println(" s");
}

/* LQR attitude controller -----------------------------------------------------
 * Model (angular_momentum_tau.mlx, reduced 6-state system A6/B6):
 *   x = [q_e1, q_e2, q_e3, omega_x, omega_y, omega_z]
 *       q_e   = vector part of the attitude error quaternion (scalar-last model,
 *               equilibrium q = [0 0 0 1])
 *       omega = body angular rate [rad/s]
 *   u = [tau_x, tau_y, tau_z]  reaction wheel torques [Nm]
 *   u = K * x   with K = -lqr(A6, B6, Q, R)   (Bryson weights, rho = 0.1)
 *
 * Paste a new K straight from MATLAB here if you retune.
 */
constexpr float LQR_K[3][6] = {
  // q_e1        q_e2        q_e3        omega_x     omega_y     omega_z
  { 11.8374307f, 0.0f,       0.0f,       1.73950634f, 0.0f,       0.0f        },  // tau_x
  { 0.0f,        11.8374307f, 0.0f,      0.0f,       1.73954037f, 0.0f        },  // tau_y
  { 0.0f,        0.0f,       11.8374307f, 0.0f,      0.0f,        1.74856749f },  // tau_z
};

// Bench-tuning multipliers on K (1.0 = MATLAB design).
//   ANGLE scales the q_e columns (stiffness), RATE scales the omega columns
//   (damping). Lowering both by the same factor keeps the shape but slows the
//   loop (more tolerant of delay); damping ratio ~ RATE / sqrt(ANGLE).
// Growing oscillations -> lower both. Overshoot but decaying -> raise RATE.
constexpr float LQR_ANGLE_GAIN_SCALE = 1.0f;
constexpr float LQR_RATE_GAIN_SCALE  = 1.0f;

// Max wheel torque used in the Bryson weighting (tau_max in MATLAB) [Nm].
constexpr float TAU_MAX_NM = 0.49f;

// Map model torque axes to ODrives (same mapping the PID used:
// odrv0 = z, odrv1 = x, odrv2 = y), and the sign between the model's tau
// (torque applied ON the wheel, positive about +body axis) and a positive
// ODrive torque command.
//
// -1 = the ODrive's positive direction points along the NEGATIVE IMU axis.
// Derived from the old PID, which was stable: it commanded
// wheel_vel = -Kp*angle, and with momentum exchange that is only stable if
// +ODrive spins the wheel about -axis. +1 here ran away (wheel spun up and the
// cube rotated with it = positive feedback).
// x/y assume the PID was also stable on those axes; check before enabling.
constexpr size_t TAU_X = 0, TAU_Y = 1, TAU_Z = 2;
constexpr size_t ODRIVE_FOR_AXIS[3]   = { 1, 2, 0 };
constexpr float  WHEEL_TORQUE_SIGN[3] = { -1.0f, -1.0f, -1.0f };

// Per-axis enable {x, y, z}. Disabled axes get zero torque.
constexpr bool   LQR_AXIS_ENABLED[3]  = { false, false, true };

// IMU data older than this disables torque output.
constexpr uint32_t ATTITUDE_STALE_TIMEOUT_MS = 50;

// Reference attitude (scalar-first {w, x, y, z}). Captured when control is
// enabled: level (roll = pitch = 0) at the current heading, so the controller
// never has to swing through a large yaw error at engagement.
float q_ref_wxyz[4] = { 1.0f, 0.0f, 0.0f, 0.0f };

float lqr_state[6]  = { 0 };
float lqr_torque[3] = { 0 };   // model-frame tau_x, tau_y, tau_z after saturation

void updateFlightState(uint32_t now) {
  static uint32_t processedSampleCount = 0;
  static float previousAccelerationMagnitude = 0.0f;
  static bool hasPreviousAccelerationSample = false;
  static uint32_t launchCandidateSince = 0;
  static bool launchCandidateActive = false;
  static uint32_t flightCandidateSince = 0;
  static bool flightCandidateActive = false;
  static uint32_t landingCandidateSince = 0;
  static bool landingCandidateActive = false;
  static bool landingDecelerationCandidateActive = false;

  if (accelerationSampleCount == 0 ||
      (uint32_t)(now - lastAccelerationMs) > ACCELERATION_STALE_TIMEOUT_MS) {
    processedSampleCount = accelerationSampleCount;
    launchCandidateActive = false;
    hasPreviousAccelerationSample = false;
    resetTrajectoryEstimate();

    if (flightState != FlightState::WAITING_FOR_LAUNCH) {
      flightState = FlightState::WAITING_FOR_LAUNCH;
      flightCandidateActive = false;
      landingCandidateActive = false;
      landingDecelerationCandidateActive = false;
      setAttitudeControlEnabled(false);
      Serial.println("IMU acceleration stale; attitude control disabled");
    }
    return;
  }

  if (processedSampleCount == accelerationSampleCount) {
    return;
  }
  processedSampleCount = accelerationSampleCount;

  accelerationMagnitude = sqrtf(
      acceleration[0] * acceleration[0] +
      acceleration[1] * acceleration[1] +
      acceleration[2] * acceleration[2]);
  const uint32_t sampleMs = lastAccelerationMs;
  const uint32_t sampleUs = lastAccelerationUs;

  // Trajectory estimate: step with the previous sample first, so a state
  // change on this sample sees the trajectory up to this sample.
  const float dt = measureLinearAcceleration(sampleUs);
  if (trajectoryIntegrating) {
    eulerStepTrajectory(dt);
  }

  switch (flightState) {
    case FlightState::WAITING_FOR_LAUNCH:
      if (accelerationMagnitude >= LAUNCH_ACCEL_THRESHOLD_M_S2) {
        if (!launchCandidateActive) {
          launchCandidateSince = sampleMs;
          launchCandidateActive = true;
          // Start integrating from rest at the origin, including the step
          // from the last pre-threshold sample to this one.
          beginTrajectory(dt);
        } else if ((uint32_t)(sampleMs - launchCandidateSince) >=
                   LAUNCH_DEBOUNCE_MS) {
          flightState = FlightState::WAITING_FOR_PROPULSION_CUTOFF;
          launchCandidateActive = false;
          launchVelocityValid = false;   // new hop confirmed: drop the previous result
        }
      } else {
        // Candidate rejected (or never started): back to rest.
        if (launchCandidateActive) {
          resetTrajectoryEstimate();
        }
        launchCandidateActive = false;
      }
      break;

    case FlightState::WAITING_FOR_PROPULSION_CUTOFF:
      if (accelerationMagnitude < PROPULSION_END_ACCEL_THRESHOLD_M_S2) {
        flightState = FlightState::AIRBORNE_WAITING_FOR_FLIGHT;
        flightCandidateActive = false;
        latchLaunchVelocityAndPredict(sampleUs, sampleMs);
        setAttitudeControlEnabled(true);
        Serial.println("Propulsion ended; attitude control enabled");
      }
      break;

    case FlightState::AIRBORNE_WAITING_FOR_FLIGHT:
      if (accelerationMagnitude <= FLIGHT_ACCEL_THRESHOLD_M_S2) {
        if (!flightCandidateActive) {
          flightCandidateSince = sampleMs;
          flightCandidateActive = true;
        } else if ((uint32_t)(sampleMs - flightCandidateSince) >=
                   FLIGHT_DEBOUNCE_MS) {
          flightState = FlightState::AIRBORNE_LANDING_ARMED;
          flightCandidateActive = false;
        }
      } else {
        flightCandidateActive = false;
      }
      break;

    case FlightState::AIRBORNE_LANDING_ARMED:
      if (accelerationMagnitude >= LANDING_ACCEL_THRESHOLD_M_S2) {
        if (landingDecelerationCandidateActive) {
          landingCandidateActive = false;
        }
        landingDecelerationCandidateActive = false;
        if (!landingCandidateActive) {
          landingCandidateSince = sampleMs;
          landingCandidateActive = true;
        } else if ((uint32_t)(sampleMs - landingCandidateSince) >=
                   LANDING_DEBOUNCE_MS) {
          flightState = FlightState::WAITING_FOR_LAUNCH;
          landingCandidateActive = false;
          endTrajectory(landingCandidateSince);
          setAttitudeControlEnabled(false);
          Serial.println("Landing detected; attitude control disabled");
        }
      } else if (landingDecelerationCandidateActive &&
                 accelerationMagnitude <= LANDING_DECEL_THRESHOLD_M_S2) {
        if ((uint32_t)(sampleMs - landingCandidateSince) >=
            LANDING_DEBOUNCE_MS) {
          flightState = FlightState::WAITING_FOR_LAUNCH;
          landingCandidateActive = false;
          landingDecelerationCandidateActive = false;
          endTrajectory(landingCandidateSince);
          setAttitudeControlEnabled(false);
          Serial.println("Landing detected; attitude control disabled");
        }
      } else if (hasPreviousAccelerationSample &&
                 previousAccelerationMagnitude > GRAVITY_M_S2 &&
                 accelerationMagnitude <= LANDING_DECEL_THRESHOLD_M_S2) {
        landingCandidateSince = sampleMs;
        landingCandidateActive = true;
        landingDecelerationCandidateActive = true;
      } else {
        landingCandidateActive = false;
        landingDecelerationCandidateActive = false;
      }
      break;
  }

  if (ballisticActive) {
    updatePredictedPosition(sampleUs);
  }
  finishTrajectorySample(sampleUs);

  previousAccelerationMagnitude = accelerationMagnitude;
  hasPreviousAccelerationSample = true;
}


// Xsens MTi: the library holds parser state and calls imu_callback()
// (defined at the bottom of this file) for every decoded data field.
void imu_callback( XsensEventFlag_t event, XsensEventData_t *mtdata );
xsens_interface_t imu_interface = XSENS_INTERFACE_RX( &imu_callback );

/* Hardware configuration ----------------------------------------------------*/
// ODrive CAN guide: https://docs.odriverobotics.com/v/latest/guides/arduino-can-guide.html

constexpr uint8_t LOW_VOLTAGE_WARNING_PIN = 5;

// CAN bus baudrate. Must match every device on the bus.
constexpr uint32_t CAN_BAUDRATE = 1000000;

// ODrive CAN node IDs
constexpr uint8_t ODRV0_NODE_ID = 1;
constexpr uint8_t ODRV1_NODE_ID = 2;
constexpr uint8_t ODRV2_NODE_ID = 3;

/* CAN (Teensy 4.1 built-in CAN1 via FlexCAN_T4) -----------------------------*/

#include <FlexCAN_T4.h>
#include "ODriveFlexCAN.hpp"
struct ODriveStatus; // hack to prevent teensy compile error

void onCanMessage(const CanMsg& msg);

FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_16> can_intf;

bool setupCan() {
  can_intf.begin();
  can_intf.setBaudRate(CAN_BAUDRATE);
  can_intf.setMaxMB(16);
  can_intf.enableFIFO();
  can_intf.enableFIFOInterrupt();
  can_intf.onReceive(onCanMessage);
  return true;
}

bool requestBusVoltage() {
  CAN_message_t request = {};
  request.id = (ODRV0_NODE_ID << 5) | Get_Bus_Voltage_Current_msg_t::cmd_id;
  request.flags.remote = true;
  request.len = 0;
  return can_intf.write(request) > 0;
}

/* ODrives -------------------------------------------------------------------*/

ODriveCAN odrv0(wrap_can_intf(can_intf), ODRV0_NODE_ID);
ODriveCAN odrv1(wrap_can_intf(can_intf), ODRV1_NODE_ID);
ODriveCAN odrv2(wrap_can_intf(can_intf), ODRV2_NODE_ID);
ODriveCAN* odrives[] = {&odrv0, &odrv1, &odrv2}; // Make sure all ODriveCAN instances are accounted for here

void commandZeroTorque() {
  for (ODriveCAN* odrive : odrives) {
    odrive->setTorque(0.0f);
  }
  for (float& tau : lqr_torque) {
    tau = 0.0f;
  }
}

// Level reference at the current heading: q_ref = yaw-only rotation. Overvej om skal sættes til 0 i stedet for yaw
void captureReferenceAttitude() {
  const float w = quaternion_wxyz[0];
  const float x = quaternion_wxyz[1];
  const float y = quaternion_wxyz[2];
  const float z = quaternion_wxyz[3];
  const float yaw = atan2f(2.0f * (w * z + x * y), 1.0f - 2.0f * (y * y + z * z));
  q_ref_wxyz[0] = cosf(0.5f * yaw);
  q_ref_wxyz[1] = 0.0f;
  q_ref_wxyz[2] = 0.0f;
  q_ref_wxyz[3] = sinf(0.5f * yaw);
}

void setAttitudeControlEnabled(bool enabled) {
  if (enabled) {
    captureReferenceAttitude();
  } else {
    commandZeroTorque();
  }
}

bool attitudeDataIsFresh(uint32_t now) {
  return receivedQuaternion && receivedGyro &&
         (uint32_t)(now - lastQuaternionMs) <= ATTITUDE_STALE_TIMEOUT_MS &&
         (uint32_t)(now - lastGyroMs) <= ATTITUDE_STALE_TIMEOUT_MS;
}

// Builds the LQR state x = [q_e(vector), omega] from the latest IMU data.
void updateLqrState() {
  // Error quaternion q_e = conj(q_ref) * q  (Hamilton product, scalar-first).
  const float rw = q_ref_wxyz[0], rx = -q_ref_wxyz[1], ry = -q_ref_wxyz[2], rz = -q_ref_wxyz[3];
  const float qw = quaternion_wxyz[0], qx = quaternion_wxyz[1], qy = quaternion_wxyz[2], qz = quaternion_wxyz[3];

  float ew = rw * qw - rx * qx - ry * qy - rz * qz;
  float ex = rw * qx + rx * qw + ry * qz - rz * qy;
  float ey = rw * qy - rx * qz + ry * qw + rz * qx;
  float ez = rw * qz + rx * qy - ry * qx + rz * qw;

  // q and -q are the same attitude; pick the short way around (scalar >= 0),
  // which matches the model's equilibrium q_4 = +1.
  if (ew < 0.0f) {
    ew = -ew; ex = -ex; ey = -ey; ez = -ez;
  }

  lqr_state[0] = ex;
  lqr_state[1] = ey;
  lqr_state[2] = ez;
  lqr_state[3] = gyro_rad_s[0];
  lqr_state[4] = gyro_rad_s[1];
  lqr_state[5] = gyro_rad_s[2];
}

// u = K * x, saturated to +-TAU_MAX_NM, then sent to the wheels.
void runLqrController() {
  updateLqrState();

  for (size_t axis = 0; axis < 3; ++axis) {
    float tau = 0.0f;
    for (size_t j = 0; j < 6 && LQR_AXIS_ENABLED[axis]; ++j) {
      const float scale = (j < 3) ? LQR_ANGLE_GAIN_SCALE : LQR_RATE_GAIN_SCALE;
      tau += scale * LQR_K[axis][j] * lqr_state[j];
    }
    lqr_torque[axis] = constrain(tau, -TAU_MAX_NM, TAU_MAX_NM);
  }

  for (size_t axis = 0; axis < 3; ++axis) {
    odrives[ODRIVE_FOR_AXIS[axis]]->setTorque(WHEEL_TORQUE_SIGN[axis] * lqr_torque[axis]);
  }
}

struct ODriveUserData {
  Heartbeat_msg_t last_heartbeat;
  bool received_heartbeat = false;
  Get_Encoder_Estimates_msg_t last_feedback;
  bool received_feedback = false;
  Get_Bus_Voltage_Current_msg_t last_bus_voltage_current;
  bool received_bus_voltage_current = false;
  uint32_t last_bus_voltage_current_ms = 0;
};

// Keep some application-specific user data for every ODrive.
ODriveUserData odrv0_user_data;
ODriveUserData odrv1_user_data;
ODriveUserData odrv2_user_data;
ODriveUserData* odrive_user_data[] = {
  &odrv0_user_data,
  &odrv1_user_data,
  &odrv2_user_data
};
const char* const odrive_names[] = {"odrv0", "odrv1", "odrv2"};

// Called every time a Heartbeat message arrives from the ODrive
void onHeartbeat(Heartbeat_msg_t& msg, void* user_data) {
  ODriveUserData* odrv_user_data = static_cast<ODriveUserData*>(user_data);
  odrv_user_data->last_heartbeat = msg;
  odrv_user_data->received_heartbeat = true;
}

// Called every time a feedback message arrives from the ODrive
void onFeedback(Get_Encoder_Estimates_msg_t& msg, void* user_data) {
  ODriveUserData* odrv_user_data = static_cast<ODriveUserData*>(user_data);
  odrv_user_data->last_feedback = msg;
  odrv_user_data->received_feedback = true;
}

void onBusVoltageCurrent(Get_Bus_Voltage_Current_msg_t& msg, void* user_data) {
  ODriveUserData* odrv_user_data = static_cast<ODriveUserData*>(user_data);
  odrv_user_data->last_bus_voltage_current = msg;
  odrv_user_data->last_bus_voltage_current_ms = millis();
  odrv_user_data->received_bus_voltage_current = true;
}

bool enableClosedLoopControl() {
  uint32_t lastStateRequestMs[3] = {};
  bool stateRequestSent[3] = {};
  uint32_t startMs = millis();

  for (size_t i = 0; i < 3; ++i) {
    if (odrive_user_data[i]->last_heartbeat.Axis_Error != 0) {
      Serial.print(odrive_names[i]);
      Serial.print(" reported axis error before enabling: 0x");
      Serial.println(odrive_user_data[i]->last_heartbeat.Axis_Error, HEX);
    }
    odrives[i]->clearErrors();
  }

  while (true) {
    bool allDrivesClosedLoop = true;
    uint32_t now = millis();

    for (size_t i = 0; i < 3; ++i) {
      if (odrive_user_data[i]->last_heartbeat.Axis_State !=
          ODriveAxisState::AXIS_STATE_CLOSED_LOOP_CONTROL) {
        allDrivesClosedLoop = false;
        if (!stateRequestSent[i] ||
            (uint32_t)(now - lastStateRequestMs[i]) >= 150) {
          odrives[i]->setState(ODriveAxisState::AXIS_STATE_CLOSED_LOOP_CONTROL);
          lastStateRequestMs[i] = now;
          stateRequestSent[i] = true;
        }
      }

      odrives[i]->setTorque(0.0f);
    }

    if (allDrivesClosedLoop) {
      return true;
    }

    if ((uint32_t)(now - startMs) >= 5000) {
      for (size_t i = 0; i < 3; ++i) {
        if (odrive_user_data[i]->last_heartbeat.Axis_State !=
            ODriveAxisState::AXIS_STATE_CLOSED_LOOP_CONTROL) {
          Serial.print("Failed to enable ");
          Serial.print(odrive_names[i]);
          Serial.print(": state=");
          Serial.print(odrive_user_data[i]->last_heartbeat.Axis_State);
          Serial.print(", axis_error=0x");
          Serial.print(odrive_user_data[i]->last_heartbeat.Axis_Error, HEX);
          Serial.print(", procedure_result=");
          Serial.println(odrive_user_data[i]->last_heartbeat.Procedure_Result);
        }
      }
      return false;
    }

    pumpEvents(can_intf);
    delay(10);
  }
}

// Called for every message that arrives on the CAN bus
void onCanMessage(const CanMsg& msg) {
  for (auto odrive: odrives) {
    onReceive(msg, *odrive);
  }
}
/*
#############
### SETUP ###
#############
*/
void setup() {
  pinMode(LOW_VOLTAGE_WARNING_PIN, OUTPUT);
  digitalWrite(LOW_VOLTAGE_WARNING_PIN, LOW);

  Serial.begin(115200);
  Serial1.begin(115200); //TTL to RS232 converter for MTI-3 IMU

  // Wait for up to 3 seconds for the serial port to be opened on the PC side.
  // If no PC connects, continue anyway.
  for (int i = 0; i < 30 && !Serial; ++i) {
    delay(100);
  }
  delay(200);


  Serial.println("Starting ODriveCAN demo");
  telemetry.printCatalog(Serial, true);

  // Register callbacks for the heartbeat and encoder feedback messages
  odrv0.onFeedback(onFeedback, &odrv0_user_data);
  odrv0.onStatus(onHeartbeat, &odrv0_user_data);
  odrv1.onFeedback(onFeedback, &odrv1_user_data);
  odrv1.onStatus(onHeartbeat, &odrv1_user_data);
  odrv2.onFeedback(onFeedback, &odrv2_user_data);
  odrv2.onStatus(onHeartbeat, &odrv2_user_data);
  odrv0.onBusVI(onBusVoltageCurrent, &odrv0_user_data);

  // Configure and initialize the CAN bus interface. This function depends on
  // your hardware and the CAN stack that you're using.
  if (!setupCan()) {
    Serial.println("CAN failed to initialize: reset required");
    while (true); // spin indefinitely
  }

  Serial.println("Waiting for ODrives...");
  while (!odrv0_user_data.received_heartbeat ||
         !odrv1_user_data.received_heartbeat ||
         !odrv2_user_data.received_heartbeat) {
    pumpEvents(can_intf);
  }

  Serial.println("found ODrives");

  // Passthrough (no ramp): the LQR model assumes the commanded torque is
  // applied instantly, and a ramp adds unmodelled lag. For ramped torque use
  // INPUT_MODE_TORQUE_RAMP + controller.config.torque_ramp_rate on the ODrive.
  Serial.println("Setting torque control mode (passthrough)...");
  for (ODriveCAN* odrive : odrives) {
    odrive->setControllerMode(ODriveControlMode::CONTROL_MODE_TORQUE_CONTROL, ODriveInputMode::INPUT_MODE_PASSTHROUGH);
  }

  Serial.println("Enabling closed loop control...");
  if (!enableClosedLoopControl()) {
    Serial.println("Startup halted; holding zero torque while CAN remains serviced.");
    while (true) {
      for (ODriveCAN* odrive : odrives) {
        odrive->setTorque(0.0f);
      }
      pumpEvents(can_intf);
      delay(10);
    }
  }

  Serial.println("ODrives running!");
}

/*
############
### LOOP ###
############
*/

void loop() {
  pumpEvents(can_intf); // This is required on some platforms to handle incoming feedback CAN messages
                        // Note that on MCP2515-based platforms, this will delay for a fixed 10ms.
                        //
                        // This has been found to reduce the number of dropped messages, however it can be removed
                        // for applications requiring loop times over 100Hz.

  uint32_t currentMillis = millis();

  static uint32_t lastBusVoltageRequestMs = 0;
  static bool busVoltageRequestFailureReported = false;
  if ((uint32_t)(currentMillis - lastBusVoltageRequestMs) >= 100) {
    lastBusVoltageRequestMs = currentMillis;
    if (!requestBusVoltage()) {
      if (!busVoltageRequestFailureReported) {
        Serial.println("Failed to send ODrive bus voltage request");
        busVoltageRequestFailureReported = true;
      }
    } else {
      busVoltageRequestFailureReported = false;
    }
  }

  static bool lowBusVoltageReported = false;
  static bool staleBusVoltageReported = false;
  const bool busVoltageIsFresh =
      odrv0_user_data.received_bus_voltage_current &&
      (uint32_t)(currentMillis - odrv0_user_data.last_bus_voltage_current_ms) <= 500;
  if (busVoltageIsFresh) {
    staleBusVoltageReported = false;
    const float busVoltage = odrv0_user_data.last_bus_voltage_current.Bus_Voltage;
    if (busVoltage < 22.5f && !lowBusVoltageReported) {
      Serial.print("WARNING: ODrive bus voltage below 22.5 V: ");
      Serial.print(busVoltage, 2);
      Serial.println(" V");
      lowBusVoltageReported = true;
      digitalWrite(LOW_VOLTAGE_WARNING_PIN, HIGH);
    }
  } else if (odrv0_user_data.received_bus_voltage_current && !staleBusVoltageReported) {
    Serial.println("WARNING: ODrive bus voltage reading is stale");
    staleBusVoltageReported = true;
  }

  //MTI PART
  //Read IMU
  while( Serial1.available() > 0 )  
  {  
      xsens_mti_parse( &imu_interface, Serial1.read() );
  }

  updateFlightState(millis());

  // Telemetry on/off commands from the dashboard / serial monitor (non-blocking)
  telemetry.poll(Serial);

  static uint32_t lastTorqueCommandMs = 0;
  uint32_t now = millis();

  //100 hz LQR control loop
  if ((uint32_t)(now - lastTorqueCommandMs) >= 10) {
    lastTorqueCommandMs = now;

    // WAITING_FOR_PROPULSION_CUTOFF: control not yet enabled (no reference
    // captured), so only the two airborne states drive the wheels.
    const bool controlActive =
        flightState == FlightState::AIRBORNE_WAITING_FOR_FLIGHT ||
        flightState == FlightState::AIRBORNE_LANDING_ARMED;

    static bool staleAttitudeReported = false;
    const bool attitudeFresh = attitudeDataIsFresh(now);

    if (controlActive && attitudeFresh) {
      staleAttitudeReported = false;
      runLqrController();
    } else {
      if (controlActive && !staleAttitudeReported) {
        Serial.println("WARNING: quaternion / rate-of-turn data stale; wheels at zero torque");
        staleAttitudeReported = true;
      }
      commandZeroTorque();
    }
  }

  // Telemetry: rate limited (default 50 Hz). Printing every channel at a high
  // rate costs loop time and the CAN line gets unhappy, so keep unneeded
  // channels off. Switch them from the dashboard instead of commenting out.
  if (telemetry.due(millis())) {
    const uint32_t nowPrint = millis();

    // -1 age means that IMU output has never been received.
    if (telemetry.on(TLM_IMU_AGE)) {
      Telemetry::value(Serial, "QuatAgeMs", receivedQuaternion ? (long)(nowPrint - lastQuaternionMs) : -1L);
      Telemetry::value(Serial, "GyroAgeMs", receivedGyro ? (long)(nowPrint - lastGyroMs) : -1L);
    }
    if (telemetry.on(TLM_LQR_TORQUE)) {
      Telemetry::value(Serial, "LQR-tau-x", lqr_torque[TAU_X]);
      Telemetry::value(Serial, "LQR-tau-y", lqr_torque[TAU_Y]);
      Telemetry::value(Serial, "LQR-tau-z", lqr_torque[TAU_Z]);
    }
    if (telemetry.on(TLM_LQR_STATE)) {
      Telemetry::value(Serial, "qe-x", lqr_state[0]);
      Telemetry::value(Serial, "qe-y", lqr_state[1]);
      Telemetry::value(Serial, "qe-z", lqr_state[2]);
      Telemetry::value(Serial, "omega-x", lqr_state[3]);
      Telemetry::value(Serial, "omega-y", lqr_state[4]);
      Telemetry::value(Serial, "omega-z", lqr_state[5]);
    }
    if (telemetry.on(TLM_FLIGHT)) {
      Telemetry::value(Serial, "AccelerationMagnitude", accelerationMagnitude, 2);
      Telemetry::value(Serial, "Airborne",
                       (flightState == FlightState::AIRBORNE_WAITING_FOR_FLIGHT ||
                        flightState == FlightState::AIRBORNE_LANDING_ARMED) ? 1L : 0L);
    }
    if (telemetry.on(TLM_EULER)) {
      Telemetry::value(Serial, "Roll", euler_pry[0], 2);
      Telemetry::value(Serial, "Pitch", euler_pry[1], 2);
      Telemetry::value(Serial, "Yaw", euler_pry[2], 2);
    }
    if (telemetry.on(TLM_QUAT)) {  // attitude, scalar-first, sensor -> local (ENU) frame
      Telemetry::value(Serial, "q-w", quaternion_wxyz[0], 5);
      Telemetry::value(Serial, "q-x", quaternion_wxyz[1], 5);
      Telemetry::value(Serial, "q-y", quaternion_wxyz[2], 5);
      Telemetry::value(Serial, "q-z", quaternion_wxyz[3], 5);
    }
    if (telemetry.on(TLM_ACCEL)) {
      Telemetry::value(Serial, "AccelX", acceleration[0], 2);
      Telemetry::value(Serial, "AccelY", acceleration[1], 2);
      Telemetry::value(Serial, "AccelZ", acceleration[2], 2);
    }
    if (telemetry.on(TLM_BUS_VOLTAGE) && busVoltageIsFresh) {
      Telemetry::value(Serial, "BusVoltage", odrv0_user_data.last_bus_voltage_current.Bus_Voltage, 2);
    }
    if (telemetry.on(TLM_WHEELS)) {
      static const char* const posNames[] = {"odrv0-pos", "odrv1-pos", "odrv2-pos"};
      static const char* const velNames[] = {"odrv0-vel", "odrv1-vel", "odrv2-vel"};
      for (size_t i = 0; i < 3; ++i) {
        if (!odrive_user_data[i]->received_feedback) continue;
        Telemetry::value(Serial, posNames[i], odrive_user_data[i]->last_feedback.Pos_Estimate);
        Telemetry::value(Serial, velNames[i], odrive_user_data[i]->last_feedback.Vel_Estimate);
      }
    }
    if (telemetry.on(TLM_BURN)) {  // integrated during the burn, ENU, origin = launch point
      Telemetry::value(Serial, "AccLinE", accel_lin_world[0], 2);
      Telemetry::value(Serial, "AccLinN", accel_lin_world[1], 2);
      Telemetry::value(Serial, "AccLinU", accel_lin_world[2], 2);
      Telemetry::value(Serial, "VelE", velocity_world[0], 3);
      Telemetry::value(Serial, "VelN", velocity_world[1], 3);
      Telemetry::value(Serial, "VelU", velocity_world[2], 3);
      Telemetry::value(Serial, "PosE", position_world[0], 3);
      Telemetry::value(Serial, "PosN", position_world[1], 3);
      Telemetry::value(Serial, "PosU", position_world[2], 3);
    }
    if (telemetry.on(TLM_BALLISTIC) && launchVelocityValid) {
      Telemetry::value(Serial, "V0E", launch_velocity[0], 3);
      Telemetry::value(Serial, "V0N", launch_velocity[1], 3);
      Telemetry::value(Serial, "V0U", launch_velocity[2], 3);
      Telemetry::value(Serial, "V0Speed", launchSpeed, 3);
      Telemetry::value(Serial, "V0Elevation", launchElevationDeg, 1);
      Telemetry::value(Serial, "V0Heading", launchHeadingDeg, 1);
      Telemetry::value(Serial, "ApexTime", ballisticApexTimeS, 3);
      Telemetry::value(Serial, "ApexHeight", ballisticApexHeightM, 3);
      Telemetry::value(Serial, "FlightTime", ballisticFlightTimeS, 3);
      Telemetry::value(Serial, "Range", ballisticRangeM, 3);
      Telemetry::value(Serial, "LandE", ballisticLanding[0], 3);
      Telemetry::value(Serial, "LandN", ballisticLanding[1], 3);
      // Estimated trajectory p_pred(t), ENU, origin = launch point
      Telemetry::value(Serial, "PredE", predicted_position[0], 3);
      Telemetry::value(Serial, "PredN", predicted_position[1], 3);
      Telemetry::value(Serial, "PredU", predicted_position[2], 3);
    }
  }
}

// Called when the library decoded an inbound packet
void imu_callback( XsensEventFlag_t event, XsensEventData_t *mtdata )
{
    switch( event )
    {
        case XSENS_EVT_QUATERNION:
            if( mtdata->type == XSENS_EVT_TYPE_FLOAT4 )
            {
                // Keep the raw quaternion (scalar-first w, x, y, z) for the LQR
                quaternion_wxyz[0] = mtdata->data.f4x4[0];
                quaternion_wxyz[1] = mtdata->data.f4x4[1];
                quaternion_wxyz[2] = mtdata->data.f4x4[2];
                quaternion_wxyz[3] = mtdata->data.f4x4[3];
                lastQuaternionMs = millis();
                receivedQuaternion = true;

                // Convert the quaternion to euler angles (telemetry 'euler' channel only)
                xsens_quaternion_to_euler( mtdata->data.f4x4, euler_pry );

                // Convert from radians to degrees
                euler_pry[0] *= (180.0 / PI);
                euler_pry[1] *= (180.0 / PI);
                euler_pry[2] *= (180.0 / PI);
            }
            break;

        case XSENS_EVT_ACCELERATION:
            if( mtdata->type == XSENS_EVT_TYPE_FLOAT3 )
            {

                acceleration[0] = mtdata->data.f4x3[0];
                acceleration[1] = mtdata->data.f4x3[1];
                acceleration[2] = mtdata->data.f4x3[2];
                lastAccelerationMs = millis();
                lastAccelerationUs = micros();
                ++accelerationSampleCount;
            }
            break;

        case XSENS_EVT_RATE_OF_TURN:
            if( mtdata->type == XSENS_EVT_TYPE_FLOAT3 )
            {
                // Body angular rate in rad/s
                gyro_rad_s[0] = mtdata->data.f4x3[0];
                gyro_rad_s[1] = mtdata->data.f4x3[1];
                gyro_rad_s[2] = mtdata->data.f4x3[2];
                lastGyroMs = millis();
                receivedGyro = true;
            }
            break;

        case XSENS_EVT_PRESSURE:
            if( mtdata->type == XSENS_EVT_TYPE_U32 )
            {
                pressure = mtdata->data.u4;
            }
            break;

        case XSENS_EVT_TEMPERATURE:
            if( mtdata->type == XSENS_EVT_TYPE_FLOAT )
            {
                temperature = mtdata->data.f4;
            }
            break;
            
        // Catches all other unused Xsens events and silences the -Wswitch warnings
        default:
            break; 
    }
}