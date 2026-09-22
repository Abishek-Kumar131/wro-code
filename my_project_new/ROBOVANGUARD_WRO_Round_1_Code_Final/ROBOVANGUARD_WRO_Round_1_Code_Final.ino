/*
  ROBOVANGUARD – WRO Future Engineers 2026
  World Robot Olympiad – Future Engineers Division

  Team ID: 1129 | Team Name: ROBOVANGUARD
  Mentor: Mr. S. Valai Ganesh (Mech, AP SG)
  Team Leader: M. Manojkumar (CSBS) – Reg. No: 953623244024
  Hardware Lead: V. Rakshit (EEE) – Reg. No: 953623105044
  Mechanical: P. Chandru (Mech) – Reg. No: 953623114009

  Camera-only architecture:
  - The Raspberry Pi 5 does all vision and decisions and sends DRIVE:<speed>:<angle>.
  - This ESP32 only drives the motor and the steering servo. No sensors are read here.
  - Failsafe: the motor stops if no command arrives for 500 ms.
  - Diagnostics sent to the Pi:
      BOOT:READY:<reset reason>   once at start-up (BROWNOUT = supply voltage dipped)
      HB:<uptime ms>              every 500 ms (uptime going backwards = the ESP32 rebooted)
      INFO:FAILSAFE_STOP          when the 500 ms failsafe stops the motor
  - Wi-Fi and Bluetooth are switched off (WRO rule 11.10).
*/

#include <WiFi.h>
#include "esp_system.h"

//#---Bot Speeds---########################################################################
int normal_speed = 250; // PWM (0-255) used by FORWARD / BACKWARD / STEER
int turn_speed = 210;   // PWM (0-255) used by LEFT / RIGHT test commands

//#---Servo Angles (+-40 deg steering range)---###########################################
int servo_center = 100;                  // 100 deg (Straight center)
int left_turn_angle = servo_center - 40; // 60 deg (Left turn)
int right_turn_angle = servo_center + 40;// 140 deg (Right turn)
//#######################################################################################

// ########### USB Serial Command & Failsafe Definitions #################################//
String serialCommandBuffer = "";
unsigned long lastCommandTime = 0;
unsigned long lastHeartbeatTime = 0;
const unsigned long COMMAND_TIMEOUT = 500;    // ms without a command -> stop motor
const unsigned long HEARTBEAT_INTERVAL = 500; // ms between HB lines
bool serialControlActive = false;

// Movement functions (in Lib_Declarations_Setup.ino)
void execute_forward();
void execute_backward();
void execute_left();
void execute_right();
void execute_stop();
void execute_steer(int angle);
void execute_drive(int speed, int angle);
void rgb_led(int r, int g, int b);

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
      markActive();
      execute_drive(speed, angle);
    } else {
      Serial.println("ERROR:INVALID_DRIVE_FORMAT");
    }
    return;
  }

  if (cmd == "FORWARD") {
    markActive();
    execute_forward();
    Serial.println("ACK:FORWARD");
  } else if (cmd == "BACKWARD") {
    markActive();
    execute_backward();
    Serial.println("ACK:BACKWARD");
  } else if (cmd == "LEFT" || cmd == "TURN_LEFT") {
    markActive();
    execute_left();
    Serial.println("ACK:LEFT");
  } else if (cmd == "RIGHT" || cmd == "TURN_RIGHT") {
    markActive();
    execute_right();
    Serial.println("ACK:RIGHT");
  } else if (cmd == "STOP") {
    execute_stop();
    serialControlActive = false;
    rgb_led(0, 0, 255); // Blue: idle, waiting for commands
    Serial.println("ACK:STOP");
  } else if (cmd == "AUTO_US_ON" || cmd == "AUTO_US_OFF") {
    Serial.println("ACK:NO_ULTRASONICS"); // accepted for older Pi scripts; nothing to switch
  } else if (cmd.startsWith("SET_TURN_DELAY:")) {
    Serial.println("ACK:SET_TURN_DELAY:UNUSED");
  } else if (cmd.startsWith("SET_SPEED:")) {
    normal_speed = constrain(cmd.substring(10).toInt(), 100, 255);
    Serial.print("ACK:SET_SPEED:");
    Serial.println(normal_speed);
  } else if (cmd.startsWith("STEER:")) {
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

// Failsafe: stop the motor if the Pi stops sending commands
void checkFailsafe() {
  if (serialControlActive && millis() - lastCommandTime > COMMAND_TIMEOUT) {
    execute_stop();
    serialControlActive = false;
    rgb_led(255, 0, 0); // Red: stopped by failsafe
    Serial.println("INFO:FAILSAFE_STOP");
  }
}

void sendHeartbeat() {
  if (millis() - lastHeartbeatTime >= HEARTBEAT_INTERVAL) {
    lastHeartbeatTime = millis();
    Serial.print("HB:");
    Serial.println(millis());
  }
}

void loop() {
  checkSerialInput();
  checkFailsafe();
  sendHeartbeat();
  delay(1);
}
