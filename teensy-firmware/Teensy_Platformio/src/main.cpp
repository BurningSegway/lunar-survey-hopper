
#include <Arduino.h>
#include "ODriveCAN.h"

//MTI includes
#include "xsens_mti.h"      // Main library
#include "xsens_utility.h"  // Needed for quaternion conversion function

//PID includes
#include "PID_v1.h"         // PID library

//MTI vars
// Cache a copy of IMU data
float    temperature     = 0;       // in degress celcius
uint32_t pressure        = 0;       // in pascals
float    euler_pry[3]    = { 0 };   // -180 to +180 degress
float    acceleration[3] = { 0 };   // in m/s^2

//PID vars
//Define Variables we'll be connecting to
double z_Setpoint, z_Input, z_Output;

//Specify the links and initial tuning parameters
double z_Kp=1.1, z_Ki=0.15, z_Kd=0.13;
PID z_PID(&z_Input, &z_Output, &z_Setpoint, z_Kp, z_Ki, z_Kd, DIRECT);

double x_Setpoint, x_Input, x_Output;
double x_Kp=1.1, x_Ki=0.15, x_Kd=0.13;
PID x_PID(&x_Input, &x_Output, &x_Setpoint, x_Kp, x_Ki, x_Kd, DIRECT);

double y_Setpoint, y_Input, y_Output;
double y_Kp=1.1, y_Ki=0.15, y_Kd=0.13;
PID y_PID(&y_Input, &y_Output, &y_Setpoint, y_Kp, y_Ki, y_Kd, DIRECT);


//MTI Callback function used by the library and such
// Callback function used by the library
void imu_callback( XsensEventFlag_t event, XsensEventData_t *mtdata );

// The library holds state and pointers to callbacks in this structure
xsens_interface_t imu_interface = XSENS_INTERFACE_RX( &imu_callback );

//Back to odrive example code
// Documentation for this example can be found here:
// https://docs.odriverobotics.com/v/latest/guides/arduino-can-guide.html


/* Configuration of example sketch -------------------------------------------*/
//Vars 

bool direction = true;

constexpr uint8_t LOW_VOLTAGE_WARNING_PIN = 5;

// CAN bus baudrate. Make sure this matches for every device on the bus
#define CAN_BAUDRATE 1000000

// ODrive node_id for odrv0
#define ODRV0_NODE_ID 01
#define ODRV1_NODE_ID 02
#define ODRV2_NODE_ID 03

// Uncomment below the line that corresponds to your hardware.
// See also "Board-specific settings" to adapt the details for your hardware setup.

#define IS_TEENSY_BUILTIN // Teensy boards with built-in CAN interface (e.g. Teensy 4.1). See below to select which interface to use.
// #define IS_ARDUINO_BUILTIN // Arduino boards with built-in CAN interface (e.g. Arduino Uno R4 Minima)
// #define IS_MCP2515 // Any board with external MCP2515 based extension module. See below to configure the module.
// #define IS_STM32_BUILTIN // STM32 boards with built-in CAN interface (e.g. STM32F4 Discovery).
// #define IS_ESP32_TWAI // ESP32 boards with built-in TWAI (CAN) interface. Directly uses the ESP-IDF TWAI driver.


/* Board-specific includes ---------------------------------------------------*/

#if defined(IS_TEENSY_BUILTIN) + defined(IS_ARDUINO_BUILTIN) + defined(IS_MCP2515) + defined(IS_STM32_BUILTIN) + defined(IS_ESP32_TWAI) != 1
#warning "Select exactly one hardware option at the top of this file."

#if CAN_HOWMANY > 0 || CANFD_HOWMANY > 0
#define IS_ARDUINO_BUILTIN
#warning "guessing that this uses HardwareCAN"
#else
#error "cannot guess hardware version"
#endif

#endif

#ifdef IS_ARDUINO_BUILTIN
// See https://github.com/arduino/ArduinoCore-API/blob/master/api/HardwareCAN.h
// and https://github.com/arduino/ArduinoCore-renesas/tree/main/libraries/Arduino_CAN

#include <Arduino_CAN.h>
#include <ODriveHardwareCAN.hpp>
#endif // IS_ARDUINO_BUILTIN

#ifdef IS_MCP2515
// See https://github.com/sandeepmistry/arduino-CAN/
#include "MCP2515.h"
#include "ODriveMCPCAN.hpp"
#endif // IS_MCP2515

#ifdef IS_TEENSY_BUILTIN
// See https://github.com/tonton81/FlexCAN_T4
// clone https://github.com/tonton81/FlexCAN_T4.git into /src
#include <FlexCAN_T4.h>
#include "ODriveFlexCAN.hpp"
struct ODriveStatus; // hack to prevent teensy compile error
#endif // IS_TEENSY_BUILTIN

#ifdef IS_STM32_BUILTIN
// See https://github.com/pazi88/STM32_CAN
#include <STM32_CAN.h>
#include "ODriveSTM32CAN.hpp"
#endif // IS_STM32_BUILTIN

#ifdef IS_ESP32_TWAI
// See https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/peripherals/twai.html
#include "driver/twai.h"
#include "ODriveESP32TWAI.hpp"
#endif // IS_ESP32_TWAI



/* Board-specific settings ---------------------------------------------------*/


/* Teensy */

#ifdef IS_TEENSY_BUILTIN

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

#endif // IS_TEENSY_BUILTIN


/* MCP2515-based extension modules -*/

#ifdef IS_MCP2515

MCP2515Class& can_intf = CAN;

// chip select pin used for the MCP2515
#define MCP2515_CS 10

// interrupt pin used for the MCP2515
// NOTE: not all Arduino pins are interruptable, check the documentation for your board!
#define MCP2515_INT 2

// freqeuncy of the crystal oscillator on the MCP2515 breakout board. 
// common values are: 16 MHz, 12 MHz, 8 MHz
#define MCP2515_CLK_HZ 8000000


static inline void receiveCallback(int packet_size) {
  if (packet_size > 8) {
    return; // not supported
  }
  CanMsg msg = {.id = (unsigned int)CAN.packetId(), .len = (uint8_t)packet_size};
  CAN.readBytes(msg.buffer, packet_size);
  onCanMessage(msg);
}

bool setupCan() {
  // configure and initialize the CAN bus interface
  CAN.setPins(MCP2515_CS, MCP2515_INT);
  CAN.setClockFrequency(MCP2515_CLK_HZ);
  if (!CAN.begin(CAN_BAUDRATE)) {
    return false;
  }

  CAN.onReceive(receiveCallback);
  return true;
}

#endif // IS_MCP2515


/* Arduinos with built-in CAN */

#ifdef IS_ARDUINO_BUILTIN

HardwareCAN& can_intf = CAN;

bool setupCan() {
  return can_intf.begin((CanBitRate)CAN_BAUDRATE);
}

#endif


/* STM32 boards with built-in CAN */

#ifdef IS_STM32_BUILTIN

STM32_CAN Can1( CAN1 );
STM32_CAN& can_intf = Can1;

bool setupCan() {
  can_intf.begin();
  can_intf.setBaudRate(CAN_BAUDRATE);
  return true;
}

#endif // IS_STM32_BUILTIN


/* ESP32 boards with built-in TWAI (CAN) */

#ifdef IS_ESP32_TWAI

// Pins used to connect to CAN bus transceiver
#define ESP32_TWAI_TX_PIN 26
#define ESP32_TWAI_RX_PIN 25

ESP32TWAIIntf can_intf;

bool setupCan() {
    twai_general_config_t g_config = TWAI_GENERAL_CONFIG_DEFAULT(
        (gpio_num_t)ESP32_TWAI_TX_PIN,
        (gpio_num_t)ESP32_TWAI_RX_PIN,
        TWAI_MODE_NORMAL
    );

    twai_timing_config_t t_config;
    switch (CAN_BAUDRATE) {
        case 1000000: t_config = TWAI_TIMING_CONFIG_1MBITS(); break;
        case 800000:  t_config = TWAI_TIMING_CONFIG_800KBITS(); break;
        case 500000:  t_config = TWAI_TIMING_CONFIG_500KBITS(); break;
        case 250000:  t_config = TWAI_TIMING_CONFIG_250KBITS(); break;
        case 125000:  t_config = TWAI_TIMING_CONFIG_125KBITS(); break;
        case 100000:  t_config = TWAI_TIMING_CONFIG_100KBITS(); break;
        case 50000:   t_config = TWAI_TIMING_CONFIG_50KBITS(); break;
        case 25000:   t_config = TWAI_TIMING_CONFIG_25KBITS(); break;
        default:      t_config = TWAI_TIMING_CONFIG_250KBITS(); break;
    }

    twai_filter_config_t f_config = TWAI_FILTER_CONFIG_ACCEPT_ALL();

    if (twai_driver_install(&g_config, &t_config, &f_config) != ESP_OK) {
        return false;
    }

    if (twai_start() != ESP_OK) {
        twai_driver_uninstall();
        return false;
    }

    return true;
}

#endif // IS_ESP32_TWAI


/* Example sketch ------------------------------------------------------------*/

// Instantiate ODrive objects
ODriveCAN odrv0(wrap_can_intf(can_intf), ODRV0_NODE_ID); // Standard CAN message ID
ODriveCAN odrv1(wrap_can_intf(can_intf), ODRV1_NODE_ID);
ODriveCAN odrv2(wrap_can_intf(can_intf), ODRV2_NODE_ID);
ODriveCAN* odrives[] = {&odrv0, &odrv1, &odrv2}; // Make sure all ODriveCAN instances are accounted for here

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

  //PID
  //initialize the variables we're linked to
  z_Input = 0;
  z_Setpoint = 45.0/2;

  x_Input = 0;
  x_Setpoint = 0;

  y_Input = 0;
  y_Setpoint = 0;

  //turn the PID on
  z_PID.SetMode(AUTOMATIC);
  z_PID.SetSampleTime(10);
  z_PID.SetOutputLimits(-10.0, 10.0);

  x_PID.SetMode(AUTOMATIC);
  x_PID.SetSampleTime(10);
  x_PID.SetOutputLimits(-10.0, 10.0);

  y_PID.SetMode(AUTOMATIC);
  y_PID.SetSampleTime(10);
  y_PID.SetOutputLimits(-10.0, 10.0);

  // Wait for up to 3 seconds for the serial port to be opened on the PC side.
  // If no PC connects, continue anyway.
  for (int i = 0; i < 30 && !Serial; ++i) {
    delay(100);
  }
  delay(200);


  Serial.println("Starting ODriveCAN demo");

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

  Serial.println("Waiting for ODrive...");
  while (!odrv0_user_data.received_heartbeat) {
    pumpEvents(can_intf);
  }

  Serial.println("found ODrive");

  Serial.println("Enabling closed loop control...");
  while (odrv0_user_data.last_heartbeat.Axis_State != ODriveAxisState::AXIS_STATE_CLOSED_LOOP_CONTROL) {
    odrv0.clearErrors();
    delay(1);
    odrv0.setState(ODriveAxisState::AXIS_STATE_CLOSED_LOOP_CONTROL);

    // Pump events for 150ms. This delay is needed for two reasons;
    // 1. If there is an error condition, such as missing DC power, the ODrive might
    //    briefly attempt to enter CLOSED_LOOP_CONTROL state, so we can't rely
    //    on the first heartbeat response, so we want to receive at least two
    //    heartbeats (100ms default interval).
    // 2. If the bus is congested, the setState command won't get through
    //    immediately but can be delayed.
    for (int i = 0; i < 15; ++i) {
      delay(10);
      pumpEvents(can_intf);
    }
  }

  Serial.println("ODrive running!");
  Serial.println("Setting ramped vel. control mode...");
  odrv0.setControllerMode(ODriveControlMode::CONTROL_MODE_VELOCITY_CONTROL, ODriveInputMode::INPUT_MODE_VEL_RAMP);
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
  static uint32_t previousMillis = 0;
  static int interval = 1000;
  float vel = 0.0;

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
    if (odrv0_user_data.last_bus_voltage_current.Bus_Voltage < 22.5f &&
        !lowBusVoltageReported) {
      Serial.println("WARNING: ODrive bus voltage below 22.5 V");
      lowBusVoltageReported = true;
      digitalWrite(LOW_VOLTAGE_WARNING_PIN, HIGH);
    }
  } else if (odrv0_user_data.received_bus_voltage_current && !staleBusVoltageReported) {
    Serial.println("WARNING: ODrive bus voltage reading is stale");
    staleBusVoltageReported = true;
  }

  uint32_t setpointMillis = millis();
  static uint32_t lastSetpoint = 0;

  if ((uint32_t)(setpointMillis - lastSetpoint) >= 5000) {
    lastSetpoint = setpointMillis;
    z_Setpoint = -z_Setpoint;

  }

  if (currentMillis - previousMillis > interval) {
    previousMillis = currentMillis;
    direction = !direction;
  }

  if (direction) {
    vel = 5.0;
  } else {
    vel = -5.0;
  }



  //MTI PART
  //Read IMU
  while( Serial1.available() > 0 )  
  {  
      xsens_mti_parse( &imu_interface, Serial1.read() );
  }

  static uint32_t lastVelocityCommandMs = 0;
  uint32_t now = millis();

  static uint32_t lastPrintMs = 0;
  uint32_t nowPrint = millis();

  double integralZoneThreshold = 5.0;

  //100 hz command rate control loop
  if ((uint32_t)(now - lastVelocityCommandMs) >= 10) {
    lastVelocityCommandMs = now;
    z_Input = euler_pry[2];
    x_Input = euler_pry[0];
    y_Input = euler_pry[1];

      // Calculate the absolute distance to the setpoint
    /*double error = abs(z_Setpoint - z_Input);

    if (error > integralZoneThreshold) {
      // Far from setpoint: Disable Ki to prevent windup and overshoot
      z_PID.SetTunings(z_Kp, 0.0, z_Kd); 
    } else {
      // Close to setpoint: Enable Ki to pull the error flat to zero
      z_PID.SetTunings(z_Kp, z_Ki, z_Kd); 
    }*/

    z_PID.Compute();
    //x_PID.Compute();
    //y_PID.Compute();

    odrv0.setVelocity(z_Output);
    //odrv1.setVelocity(x_Output);
    //odrv2.setVelocity(y_Output);
  }

  // print position and velocity for Serial Plotter - maybe should be put in a callback function or whatever, and the prints into the rate limited print
  if (odrv0_user_data.received_feedback) {
    Get_Encoder_Estimates_msg_t feedback = odrv0_user_data.last_feedback;
    odrv0_user_data.received_feedback = false;
    Serial.print(">odrv0-pos:");
    Serial.println(feedback.Pos_Estimate);
    Serial.print(">odrv0-vel:");
    Serial.println(feedback.Vel_Estimate);
  }

  //50 hz print rate for serial plotter otherwise bogs system down, and the CAN line gets unhappy :(
  if ((uint32_t)(nowPrint - lastPrintMs) >= 20) {
    lastPrintMs = nowPrint;

    Serial.print(">PID-Output-z:");
    Serial.println(z_Output);
    Serial.print(">PID-Output-x:");
    Serial.println(x_Output);
    Serial.print(">PID-Output-y:");
    Serial.println(y_Output);


    Serial.print(">Roll:");
    Serial.println(euler_pry[0]);
    Serial.print(">Pitch:");
    Serial.println(euler_pry[1]);
    Serial.print(">Yaw:");
    Serial.println(euler_pry[2]);

    Serial.print(">Yaw-reference:");
    Serial.println(z_Setpoint);
    Serial.print(">Roll-reference:");
    Serial.println(x_Setpoint);
    Serial.print(">Pitch-reference:");
    Serial.println(y_Setpoint);

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
                // Convert the quaternion to euler angles
                xsens_quaternion_to_euler( mtdata->data.f4x4, euler_pry );

                // Convert from radians to degrees
                euler_pry[0] *= (180.0 / PI);
                euler_pry[1] *= (180.0 / PI);
                euler_pry[2] *= (180.0 / PI);

                // Output to PC formatted for the Arduino Serial Plotter
                /*Serial.print(">Roll:");
                Serial.println(euler_pry[0]);
                Serial.print(">Pitch:");
                Serial.println(euler_pry[1]);
                Serial.print(">Yaw:");
                Serial.println(euler_pry[2]); */
            }
            break;

        case XSENS_EVT_ACCELERATION:
            if( mtdata->type == XSENS_EVT_TYPE_FLOAT3 )
            {

                acceleration[0] = mtdata->data.f4x3[0];
                acceleration[1] = mtdata->data.f4x3[1];
                acceleration[2] = mtdata->data.f4x3[2];

                /*Serial.print(">AccelX:");
                Serial.println(acceleration[0]);
                Serial.print(">AccelY:");
                Serial.println(acceleration[1]);
                Serial.print(">AccelZ:");
                Serial.println(acceleration[2]);*/
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