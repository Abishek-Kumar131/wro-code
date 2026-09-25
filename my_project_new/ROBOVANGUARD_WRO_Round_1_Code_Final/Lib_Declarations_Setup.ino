/* BRANCH: us-vision-hybrid  -  camera + 6 ultrasonic sensors (ESP32 pings one sensor per cycle) */
/*
  ROBOVANGUARD – WRO Future Engineers 2026
  World Robot Olympiad – Future Engineers Division

  Team ID: 1129 | Team Name: ROBOVANGUARD
  Mentor: Mr. S. Valai Ganesh (Mech, AP SG)
  Team Leader: M. Manojkumar (CSBS) – Reg. No: 953623244024
  Hardware Lead: V. Rakshit (EEE) – Reg. No: 953623105044
  Mechanical: P. Chandru (Mech) – Reg. No: 953623114009

  File: Lib_Declarations_Setup.ino (Round 1)
  Purpose: Library includes, pin/IO declarations, helper functions, and setup()
           used by the Round 1 control logic.
*/

#include <Wire.h>
#include <NewPing.h>
#include <FastLED.h>

// ########### Declerations ############################################################################################################ //

// RGB Led
#define LED_PIN 15
#define NUM_LEDS 1
CRGB leds[NUM_LEDS];

// DC Motor
const int motorPin1 = 32; 
const int motorPin2 = 33; 

const int dc_chan1 = 2;
const int dc_chan2 = 3;
const int nslp = 13; 
const int frequency = 20000; // 20kHz inaudible ultrasonic PWM frequency (eliminates coil whine)


// Servo Motor (Hardware LEDC PWM - Channel 4, Timer 2 @ 50Hz)
#define SERVO_PIN 27
const int servo_chan = 4;     // Dedicated LEDC Channel 4 (isolated from DC motor channels 2 & 3)
const int servo_freq = 50;    // Standard 50Hz servo refresh rate (20ms period)
const int servo_res = 14;     // 14-bit resolution (0 - 16383)

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

// 200 cm is more than enough on a 3 m track and caps one ping at ~11 ms of blocking
// (400 cm made every ping block up to ~23 ms).
#define MAX_DISTANCE 200

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


// Servo Functions using Hardware LEDC (50Hz, 14-bit)
void moveServoTo(int angle) {
  // Hard limit of the steering linkage: 75 to 125 deg (100 = straight).
  // Nothing past this is reachable, so clamp here as well as on the Pi.
  angle = constrain(angle, 75, 125);
  // Map angle (0 - 180 deg) to standard servo pulse width (500us to 2400us)
  long pulse_us = map(angle, 0, 180, 500, 2400);
  // Convert pulse width (in microseconds) to 14-bit duty cycle at 50Hz (20,000us period):
  // duty = (pulse_us * 16383) / 20000
  long duty = (pulse_us * 16383) / 20000;

#if defined(ESP_ARDUINO_VERSION_MAJOR) && (ESP_ARDUINO_VERSION_MAJOR >= 3)
  ledcWrite(SERVO_PIN, duty);
#else
  ledcWrite(servo_chan, duty);
#endif
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

void execute_steer(int angle) {
  moveServoTo(angle);
  motor_forward(normal_speed);
}

void execute_drive(int speed, int angle) {
  moveServoTo(angle);
  if (speed > 0) {
    motor_forward(speed);
  } else if (speed < 0) {
    motor_backward(-speed);
  } else {
    motor_stop();
  }
}



// ########### Ultrasonic round-robin ###################################################
// One sensor per call, at most one ping every PING_INTERVAL ms. All six refresh in
// ~120 ms. The order alternates front/side/back so that two sensors pointing in similar
// directions never fire back to back, which is what caused ghost echoes before.

extern int f_us, f1_us, f2_us, b_us, l_us, r_us;

const unsigned long PING_INTERVAL = 20;   // ms between pings
unsigned long lastPingTime = 0;
int pingIndex = 0;

void updateUltrasonics() {
  if (millis() - lastPingTime < PING_INTERVAL) return;
  lastPingTime = millis();

  switch (pingIndex) {
    case 0: f_us  = sonar1.ping_cm(); break;   // front
    case 1: l_us  = sonar3.ping_cm(); break;   // left
    case 2: f1_us = sonar5.ping_cm(); break;   // front-left
    case 3: r_us  = sonar4.ping_cm(); break;   // right
    case 4: f2_us = sonar6.ping_cm(); break;   // front-right
    case 5: b_us  = sonar2.ping_cm(); break;   // back
  }
  pingIndex = (pingIndex + 1) % 6;
}

// ########### Setup ############################################################################################################ //
void setup() {
  Serial.begin(115200);

  //######### Radios off (WRO rule 11.10: no wireless during rounds) #########//
  WiFi.mode(WIFI_OFF);
  btStop();

  //######### RGB Led Setup #########//
  FastLED.addLeds<NEOPIXEL, LED_PIN>(leds, NUM_LEDS);
  FastLED.clear();
  FastLED.show();
  
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
  motor_stop(); // Ensure motors start in completely silent stopped state

  //######### Servo Motor Setup (Dedicated Hardware LEDC Channel 4 @ 50Hz) ###########//
#if defined(ESP_ARDUINO_VERSION_MAJOR) && (ESP_ARDUINO_VERSION_MAJOR >= 3)
  ledcAttach(SERVO_PIN, servo_freq, servo_res);
#else
  ledcSetup(servo_chan, servo_freq, servo_res);
  ledcAttachPin(SERVO_PIN, servo_chan);
#endif

  // No boot wiggle: if the ESP32 ever resets mid-run, the wheels must not jerk
  moveServoTo(servo_center);

  rgb_led(0, 0, 255); // Blue: idle, waiting for Pi commands
  Serial.print("BOOT:READY:");
  Serial.println(resetReasonName(esp_reset_reason()));
  lastHeartbeatTime = millis();
}
