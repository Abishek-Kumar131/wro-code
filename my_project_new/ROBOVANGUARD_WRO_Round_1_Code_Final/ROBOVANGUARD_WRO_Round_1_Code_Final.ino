/* SHARED FIRMWARE - byte-identical on every branch.
   Flash it once. Switching the Pi between branches needs NO re-flash. */
/*
  ROBOVANGUARD – WRO Future Engineers 2026
  World Robot Olympiad – Future Engineers Division

  Team ID: 1129 | Team Name: ROBOVANGUARD
  Mentor: Mr. S. Valai Ganesh (Mech, AP SG)
  Team Leader: M. Manojkumar (CSBS) – Reg. No: 953623244024
  Hardware Lead: V. Rakshit (EEE) – Reg. No: 953623105044
  Mechanical: P. Chandru (Mech) – Reg. No: 953623114009

  The Pi 5 does all the vision and decisions and sends DRIVE:<speed>:<angle>.
  This ESP32 only drives the motor and steering servo, and reports what it senses.

  Ultrasonic sensors are OPTIONAL. This firmware always reads them and always streams
  the US: line, so both Pi branches work against the same binary:
    - us-vision-hybrid          uses the readings
    - camera-only-canada-strategy  ignores them
  With no sensors connected, every reading simply stays 0 ("nothing in range") and
  nothing else changes.

  - ONE sensor is pinged per cycle, 20 ms apart (fixed: pinging all six back to back
    blocked the loop for up to 140 ms and made the readings cross-talk).
    All six refresh every ~120 ms and the loop never blocks for more than one ping.
  - Failsafe: the motor stops if no command arrives for 500 ms (applies during timed
    arc turns too).
  - Steering is limited to 75-125 degrees, the mechanical range of this linkage.
  - Telemetry to the Pi:
      US:F:..,F1:..,F2:..,L:..,R:..,B:..   every 100 ms (0 = nothing in range)
      BOOT:READY:<reset reason>           once at start-up (BROWNOUT = supply dipped)
      HB:<uptime ms>                      every 500 ms (uptime going back = it rebooted)
      INFO:FAILSAFE_STOP                  when the failsafe stops the motor
  - Wi-Fi and Bluetooth are switched off (WRO rule 11.10).

  Requires the NewPing and FastLED libraries in the Arduino IDE.
*/

#include <WiFi.h>
#include "esp_system.h"

//#---Bot Speeds & Timings---#############################################################
int normal_speed = 250; // PWM (0-255) straightaway speed for FORWARD / STEER
int turn_speed = 210;   // PWM (0-255) during LEFT / RIGHT arc turns
int turn_delay = 2000;  // ms (corner arc duration for the LEFT/RIGHT commands)
int fus_slow_speed = 225; // PWM when a wall is close ahead (AUTO_US_ON mode only)
int fus_slow_dist = 80;   // cm front distance that triggers that slowdown

//#---Servo Angles (+-25 deg mechanical steering range)---###############################
int servo_center = 100;                  // 100 deg (Straight center)
int left_turn_angle = servo_center - 25; // 75 deg (full left)
int right_turn_angle = servo_center + 25;// 125 deg (full right)
int target_wall_dist = 25;               // cm target distance from a side wall
//#######################################################################################

int f_us, f1_us, f2_us, b_us, l_us, r_us;

// ########### USB Serial Command & Failsafe Definitions #################################//
String serialCommandBuffer = "";
unsigned long lastCommandTime = 0;
unsigned long lastTelemetryTime = 0;
unsigned long lastHeartbeatTime = 0;
const unsigned long COMMAND_TIMEOUT = 500;
const unsigned long TELEMETRY_INTERVAL = 100;
const unsigned long HEARTBEAT_INTERVAL = 500;
bool serialControlActive = false;
bool useSideUltrasonic = false;   // ESP32-side wall centering; off while the Pi sends DRIVE

// Timed Arc Turn State Machine (used by the LEFT / RIGHT commands and the test utility)
bool isTurning = false;
bool last_cmd_was_left = false;
unsigned long turnStartTime = 0;

// Functions defined in Lib_Declarations_Setup.ino
void execute_forward();
void execute_backward();
void execute_left();
void execute_right();
void execute_stop();
void execute_steer(int angle);
void execute_drive(int speed, int angle);
void updateUltrasonics();
void rgb_led(int r, int g, int b);
void moveServoTo(int angle);
void motor_forward(int speed);

const char* resetReasonName(esp_reset_reason_t reason) {
  switch (reason) {
    case ESP_RST_POWERON:   return "POWERON";
    case ESP_RST_EXT:       return "EXTERNAL_PIN";
    case ESP_RST_SW:        return "SOFTWARE";
    case ESP_RST_PANIC:     return "PANIC";
    case ESP_RST_INT_WDT:   return "INT_WATCHDOG";
    case ESP_RST_TASK_WDT:  return "TASK_WATCHDOG";
    case ESP_RST_WDT:       return "WATCHDOG";
    case ESP_RST_DEEPSLEEP: return "DEEPSLEEP";
    case ESP_RST_BROWNOUT:  return "BROWNOUT";
    case ESP_RST_SDIO:      return "SDIO";
    default:                return "UNKNOWN";
  }
}

void markActive() {
  lastCommandTime = millis();
  if (!serialControlActive) {
    serialControlActive = true;
    rgb_led(0, 255, 0); // Green: driving under Pi control
  }
}

// Process incoming command from Raspberry Pi 5 over USB Serial
void processCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();
  if (cmd.length() == 0) return;

  if (cmd.startsWith("DRIVE:")) {
    // Hot path: sent up to ~30x per second, so no ACK is printed
    int firstColon = cmd.indexOf(':');
    int secondColon = cmd.indexOf(':', firstColon + 1);
    if (secondColon != -1) {
      int speed = cmd.substring(firstColon + 1, secondColon).toInt();
      int angle = cmd.substring(secondColon + 1).toInt();
      isTurning = false;
      useSideUltrasonic = false;
      markActive();
      execute_drive(speed, angle);
    } else {
      Serial.println("ERROR:INVALID_DRIVE_FORMAT");
    }
    return;
  }

  if (cmd == "FORWARD") {
    isTurning = false;
    markActive();
    execute_forward();
    Serial.println("ACK:FORWARD");
  } else if (cmd == "BACKWARD") {
    isTurning = false;
    markActive();
    execute_backward();
    Serial.println("ACK:BACKWARD");
  } else if (cmd == "LEFT" || cmd == "TURN_LEFT") {
    last_cmd_was_left = true;
    markActive();
    if (!isTurning) {
      isTurning = true;
      turnStartTime = millis();
    }
    execute_left();
    Serial.println("ACK:LEFT");
  } else if (cmd == "RIGHT" || cmd == "TURN_RIGHT") {
    last_cmd_was_left = false;
    markActive();
    if (!isTurning) {
      isTurning = true;
      turnStartTime = millis();
    }
    execute_right();
    Serial.println("ACK:RIGHT");
  } else if (cmd == "STOP") {
    isTurning = false;
    execute_stop();
    serialControlActive = false;
    rgb_led(0, 0, 255); // Blue: idle
    Serial.println("ACK:STOP");
  } else if (cmd == "AUTO_US_ON") {
    useSideUltrasonic = true;
    markActive();
    Serial.println("ACK:AUTO_US_ON");
  } else if (cmd == "AUTO_US_OFF") {
    useSideUltrasonic = false;
    Serial.println("ACK:AUTO_US_OFF");
  } else if (cmd.startsWith("SET_TURN_DELAY:")) {
    turn_delay = cmd.substring(15).toInt();
    Serial.print("ACK:SET_TURN_DELAY:");
    Serial.println(turn_delay);
  } else if (cmd.startsWith("SET_SPEED:")) {
    normal_speed = constrain(cmd.substring(10).toInt(), 100, 255);
    Serial.print("ACK:SET_SPEED:");
    Serial.println(normal_speed);
  } else if (cmd.startsWith("STEER:")) {
    isTurning = false;
    useSideUltrasonic = false;
    int angle = cmd.substring(6).toInt();
    markActive();
    execute_steer(angle);
    Serial.print("ACK:STEER:");
    Serial.println(angle);
  } else {
    Serial.print("ERROR:UNKNOWN_COMMAND:");
    Serial.println(cmd);
  }
}

// Non-blocking serial character receiver
void checkSerialInput() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (serialCommandBuffer.length() > 0) {
        processCommand(serialCommandBuffer);
        serialCommandBuffer = "";
      }
    } else if (serialCommandBuffer.length() < 64) {
      serialCommandBuffer += c;
    } else {
      serialCommandBuffer = ""; // overlong garbage line: drop it
    }
  }
}

// Failsafe: stop the motor if the Pi stops sending commands.
// Applies during arc turns too, so a lost link can never leave the car turning.
void checkFailsafe() {
  if (serialControlActive && millis() - lastCommandTime > COMMAND_TIMEOUT) {
    isTurning = false;
    execute_stop();
    serialControlActive = false;
    rgb_led(255, 0, 0); // Red: stopped by failsafe
    Serial.println("INFO:FAILSAFE_STOP");
  }
}

void sendUltrasonicTelemetry() {
  if (millis() - lastTelemetryTime >= TELEMETRY_INTERVAL) {
    lastTelemetryTime = millis();
    Serial.print("US:F:");   Serial.print(f_us);
    Serial.print(",F1:");    Serial.print(f1_us);
    Serial.print(",F2:");    Serial.print(f2_us);
    Serial.print(",L:");     Serial.print(l_us);
    Serial.print(",R:");     Serial.print(r_us);
    Serial.print(",B:");     Serial.println(b_us);
  }
}

void sendHeartbeat() {
  if (millis() - lastHeartbeatTime >= HEARTBEAT_INTERVAL) {
    lastHeartbeatTime = millis();
    Serial.print("HB:");
    Serial.println(millis());
  }
}

// ESP32-side wall centering, only while AUTO_US_ON is active (any DRIVE turns it off).
// Treats 0 as "nothing in range", never as "a wall is touching us".
void side_us_logic_fun() {
  if (f_us > 0 && f_us < fus_slow_dist) {
    motor_forward(fus_slow_speed);
  } else {
    motor_forward(normal_speed);
  }

  bool valid_left = (l_us > 5 && l_us < 120);
  bool valid_right = (r_us > 5 && r_us < 120);

  if (valid_left && valid_right) {
    moveServoTo(servo_center + (r_us - l_us) * 2);   // steer away from the closer wall
  } else if (valid_left) {
    moveServoTo(servo_center + (target_wall_dist - l_us) * 2);
  } else if (valid_right) {
    moveServoTo(servo_center - (target_wall_dist - r_us) * 2);
  } else {
    moveServoTo(servo_center);
  }
}

void loop() {
  updateUltrasonics();      // one sensor per cycle, never all six at once
  checkSerialInput();
  checkFailsafe();
  sendUltrasonicTelemetry();
  sendHeartbeat();

  if (isTurning) {
    if (millis() - turnStartTime >= (unsigned long)turn_delay) {
      isTurning = false;
      moveServoTo(servo_center);
      motor_forward(normal_speed);
      Serial.println("ACK:TURN_COMPLETE");
    }
  } else if (serialControlActive && useSideUltrasonic) {
    side_us_logic_fun();
  }
}
