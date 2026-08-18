/*
  ROBOVANGUARD – WRO Future Engineers 2025
  World Robot Olympiad – Future Engineers Division

  Team ID: 1129 | Team Name: ROBOVANGUARD
  Mentor: Mr. S. Valai Ganesh (Mech, AP SG)
  Team Leader: M. Manojkumar (CSBS) – Reg. No: 953623244024
  Hardware Lead: V. Rakshit (EEE) – Reg. No: 953623105044
  Mechanical: P. Chandru (Mech) – Reg. No: 953623114009

  File: Lib_Declarations_Setup.ino (Round 1)
  Purpose: Library includes, pin/IO declarations, helper functions, and setup()
           used by the Round 1 control logic.

  Notes:
  - This header is documentation-only to clarify authorship and structure.
  - No functional behavior is modified.
*/

#include <Wire.h>
#include <Adafruit_TCS34725.h>
#include <ESP32Servo.h>
#include <NewPing.h>
#include <FastLED.h>

// ########### Declerations ############################################################################################################ //

// RGB Led
#define LED_PIN 15
#define NUM_LEDS 1
CRGB leds[NUM_LEDS];

// Colour Sensor
#define TCS3414CS_ADDRESS 0x29 //ColorSensor address 0x29

// INTEGRATIONTIME - 2.4 - 614 ms | GAIN - 1x, 4x, 16x, 64x.
Adafruit_TCS34725 tcs1 = Adafruit_TCS34725(TCS34725_INTEGRATIONTIME_2_4MS, TCS34725_GAIN_4X); // Initializing ColorSensor 

int col_out;

// DC Motor
const int motorPin1 = 32; 
const int motorPin2 = 33; 

const int dc_chan1 = 2;
const int dc_chan2 = 3;
const int nslp = 13; 
const int frequency = 5000;


// Servo Motor
#define SERVO_PIN 27
Servo servo;

// Ultrasonic Sensors

#define FRONT_TRIGGER 12 
#define FRONT_ECHO  4  

#define FRONT1_TRIGGER 16
#define FRONT1_ECHO 14

#define FRONT2_TRIGGER 25 
#define FRONT2_ECHO  26  

#define BACK_TRIGGER 17
#define BACK_ECHO 19

#define LEFT_TRIGGER  2
#define LEFT_ECHO  23

#define RIGHT_TRIGGER 5
#define RIGHT_ECHO  18

#define MAX_DISTANCE 400

NewPing sonar1(FRONT_TRIGGER, FRONT_ECHO, MAX_DISTANCE); 
NewPing sonar5(FRONT1_TRIGGER, FRONT1_ECHO, MAX_DISTANCE); 
NewPing sonar6(FRONT2_TRIGGER, FRONT2_ECHO, MAX_DISTANCE); 

NewPing sonar2(BACK_TRIGGER, BACK_ECHO, MAX_DISTANCE); 
NewPing sonar3(LEFT_TRIGGER, LEFT_ECHO, MAX_DISTANCE);
NewPing sonar4(RIGHT_TRIGGER, RIGHT_ECHO, MAX_DISTANCE); 



// ########### Functions ############################################################################################################ //
// RGB Led Function
void rgb_led(int r, int g, int b)
{
  leds[0] = CRGB(r, g, b);
  FastLED.show();
}

// ColorSensor Function

int front_colour_sensor() {
  //TCA9548A(0);
  uint16_t r, g, b, c;
  tcs1.getRawData(&r, &g, &b, &c);
  uint16_t colorTemp = tcs1.calculateColorTemperature(r, g, b);
  //Serial.println("CS-1 Color Temp: "+ String(colorTemp));
  //uint16_t lux = tcs1.calculateLux(r, g, b);
  //Serial.println("Red: "+ String(r)+" Green: "+String(g)+" Blue: "+String(b)+" Clear: "+String(c));
  

  // Color Temp Condition
  if((0 < colorTemp) && (colorTemp < orange_line_thold) && (colorTemp != 0)) // For Orange Line
  {col_out = 1; } 

  else if((colorTemp > blue_line_thold) && (colorTemp != 0)) // For Blue Line 
  {col_out = 3;}
  
  else
  {col_out = 0;}

  return col_out;
}



// DC Motor Functions (Compatible with both ESP32 Core 2.x and Core 3.x)
#if defined(ESP_ARDUINO_VERSION_MAJOR) && (ESP_ARDUINO_VERSION_MAJOR >= 3)
void motor_forward(int speed) {
  ledcWrite(motorPin1, speed);
  ledcWrite(motorPin2, 0);
}

void motor_backward(int speed) {
  ledcWrite(motorPin1, 0);
  ledcWrite(motorPin2, speed);
}

void motor_stop() {
  ledcWrite(motorPin1, 0);
  ledcWrite(motorPin2, 0);
}
#else
void motor_forward(int speed) {
  ledcWrite(dc_chan1, speed);
  ledcWrite(dc_chan2, 0);
}

void motor_backward(int speed) {
  ledcWrite(dc_chan1, 0);
  ledcWrite(dc_chan2, speed);
}

void motor_stop() {
  ledcWrite(dc_chan1, 0);
  ledcWrite(dc_chan2, 0);
}
#endif


// Servo Functions
void moveServoTo(int angle) {
  // Constrain the angle between 0 and 180 degrees
  angle = constrain(angle, 75, 125);
  servo.write(angle);
  //delay(15);
  //Serial.println("Servo Angle : "+String(angle));
}

// Forward declarations for speeds and angles defined in main ino
extern int normal_speed;
extern int turn_speed;
extern int servo_center;
extern int left_turn_angle;
extern int right_turn_angle;

// Unified Movement Execution Functions (Used by USB Serial & Navigation Logic)
void execute_forward() {
  moveServoTo(servo_center);
  motor_forward(normal_speed);
}

void execute_backward() {
  moveServoTo(servo_center);
  motor_backward(normal_speed);
}

void execute_left() {
  moveServoTo(left_turn_angle);
  motor_forward(turn_speed);
}

void execute_right() {
  moveServoTo(right_turn_angle);
  motor_forward(turn_speed);
}

void execute_stop() {
  motor_stop();
  moveServoTo(servo_center);
}


// UltraSonic Function

void US_Values(int &f, int &f1, int &f2, int &b, int &l, int &r)
{
  unsigned int front_us = sonar1.ping_cm();
  unsigned int front1_us = sonar5.ping_cm();
  unsigned int front2_us = sonar6.ping_cm();
  unsigned int back_us = sonar2.ping_cm(); 
  unsigned int left_us = sonar3.ping_cm(); 
  unsigned int right_us = sonar4.ping_cm(); 

  f = front_us;
  f1 = front1_us;
  f2 = front2_us;
  b = back_us;
  l = left_us;
  r = right_us;
}

// ########### Setup ############################################################################################################ //
void setup() {
  Serial.begin(115200);

  //######### DPDT Setup #########//
  pinMode(DPDT_Push_Button_Pin, INPUT);

  //######### RGB Led Setup #########//
  FastLED.addLeds<NEOPIXEL, LED_PIN>(leds, NUM_LEDS);
  FastLED.clear();
  FastLED.show();

  //######### Colour Sensor Setup #########//
  analogReadResolution(12);       
  analogSetAttenuation(ADC_0db);
  
  //######### DC Motor Setup ###########//
#if defined(ESP_ARDUINO_VERSION_MAJOR) && (ESP_ARDUINO_VERSION_MAJOR >= 3)
  ledcAttach(motorPin1, frequency, 8);
  ledcAttach(motorPin2, frequency, 8);
#else
  ledcSetup(dc_chan1, frequency, 8);
  ledcSetup(dc_chan2, frequency, 8);
  ledcAttachPin(motorPin1, dc_chan1);
  ledcAttachPin(motorPin2, dc_chan2);
#endif
  pinMode(nslp, OUTPUT);
  digitalWrite(nslp, HIGH);

  //######### Servo Motor Setup ###########//

  servo.attach(SERVO_PIN, 500, 2400);
  //initial servo angle
  moveServoTo(servo_center);
  delay(500);
  }
