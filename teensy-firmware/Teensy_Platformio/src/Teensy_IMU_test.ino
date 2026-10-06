#include <Arduino.h>
#include "xsens_mti.h"      // Main library
#include "xsens_utility.h"  // Needed for quaternion conversion function

// Cache a copy of IMU data
float    temperature     = 0;       // in degress celcius
uint32_t pressure        = 0;       // in pascals
float    euler_pry[3]    = { 0 };   // -180 to +180 degress
float    acceleration[3] = { 0 };   // in m/s^2

// Callback function used by the library
void imu_callback( XsensEventFlag_t event, XsensEventData_t *mtdata );

// The library holds state and pointers to callbacks in this structure
xsens_interface_t imu_interface = XSENS_INTERFACE_RX( &imu_callback );

#ifndef LED_BUILTIN
    #define LED_BUILTIN 13
#endif

void setup( void )
{
    // Start USB serial for PC Serial Plotter output
    Serial.begin( 115200 );
    
    // Start Hardware Serial1 for the Xsens IMU on Teensy pins 0 (RX1) and 1 (TX1)
    Serial1.begin( 115200 );
    
    pinMode( LED_BUILTIN, OUTPUT );
}

void loop( void )
{
    // Read from Hardware Serial1 instead of USB Serial
    while( Serial1.available() > 0 )  
    {  
        xsens_mti_parse( &imu_interface, Serial1.read() );
    }

    // Light goes high if the IMU pitch exceeds 10 degrees
    digitalWrite( LED_BUILTIN, (euler_pry[0] > 10.0f) );
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
                Serial.print("Roll:");
                Serial.print(euler_pry[0]);
                Serial.print(",");
                Serial.print("Pitch:");
                Serial.print(euler_pry[1]);
                Serial.print(",");
                Serial.print("Yaw:");
                Serial.println(euler_pry[2]);
            }
            break;

        case XSENS_EVT_ACCELERATION:
            if( mtdata->type == XSENS_EVT_TYPE_FLOAT3 )
            {
                acceleration[0] = mtdata->data.f4x3[0];
                acceleration[1] = mtdata->data.f4x3[1];
                acceleration[2] = mtdata->data.f4x3[2];
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