/*
  ROBOVANGUARD – WRO Future Engineers 2025
  World Robot Olympiad – Future Engineers Division

  Team ID: 1129
  Team Name: ROBOVANGUARD

  Mentor: Mr. S. Valai Ganesh (Mech, AP SG)
  Team Leader: M. Manojkumar (CSBS) – Reg. No: 953623244024
  Hardware Lead: V. Rakshit (EEE) – Reg. No: 953623105044
  Mechanical: P. Chandru (Mech) – Reg. No: 953623114009

  Project Summary:
  - Intelligent autonomous vehicle utilizing ESP32, Ackermann steering, and ultrasonic array.
  - Controls drive motors and steering servo via USB Serial from Raspberry Pi 5 or standalone ultrasonic logic.
*/

// ########### Configuration & Navigation Parameters ################################################## //
int line_chk_count = 12;  // Lap check threshold
int line_count = 0;

//#---Bot Speeds---######################################################################
int normal_speed = 200; // pwm (0-255)
int turn_speed = 220;   // pwm (0-255)
int turn_delay = 2000;  // ms
//#######################################################################################
int fus_slow_speed = 200; // pwm
int fus_slow_dist = 130;  // cm

//#---Servo Angles---####################################################################
int servo_center = 100;                 // 100 deg (Straight center)
int left_turn_angle = servo_center - 20; // 80 deg (Left turn)
int right_turn_angle = servo_center + 20;// 120 deg (Right turn)
//#######################################################################################

bool lt_st_count = 0;
bool rt_st_count = 0;
bool left_right_arc_turn = 1;
bool left_right_r_turn = 0;

#define DPDT_Push_Button_Pin 34

int f_us, f1_us, f2_us, b_us, l_us, r_us, fusa, far;

bool LOGIC_LOCK = 1; // 1 True state.
bool DPDT_STATE = 0; // 0 False state.

// ########### USB Serial Command & Failsafe Definitions #################################//
String serialCommandBuffer = "";
unsigned long lastCommandTime = 0;
const unsigned long COMMAND_TIMEOUT = 500; // 500ms failsafe timeout
bool serialControlActive = false;

// Forward declarations for unified movement execution functions (in Lib_Declarations_Setup.ino)
void execute_forward();
void execute_backward();
void execute_left();
void execute_right();
void execute_stop();
void execute_steer(int angle);
void execute_drive(int speed, int angle);

// Process incoming command from Raspberry Pi 5 over USB Serial
void processCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();
  if (cmd.length() == 0) return;

  lastCommandTime = millis();
  serialControlActive = true;

  if (cmd == "FORWARD") {
    execute_forward();
    Serial.println("ACK:FORWARD");
  } else if (cmd == "BACKWARD") {
    execute_backward();
    Serial.println("ACK:BACKWARD");
  } else if (cmd == "LEFT") {
    execute_left();
    Serial.println("ACK:LEFT");
  } else if (cmd == "RIGHT") {
    execute_right();
    Serial.println("ACK:RIGHT");
  } else if (cmd == "STOP") {
    execute_stop();
    Serial.println("ACK:STOP");
  } else if (cmd.startsWith("STEER:")) {
    int angle = cmd.substring(6).toInt();
    execute_steer(angle);
    Serial.print("ACK:STEER:");
    Serial.println(angle);
  } else if (cmd.startsWith("DRIVE:")) {
    int firstColon = cmd.indexOf(':');
    int secondColon = cmd.indexOf(':', firstColon + 1);
    if (secondColon != -1) {
      int speed = cmd.substring(firstColon + 1, secondColon).toInt();
      int angle = cmd.substring(secondColon + 1).toInt();
      execute_drive(speed, angle);
      Serial.print("ACK:DRIVE:");
      Serial.print(speed);
      Serial.print(":");
      Serial.println(angle);
    } else {
      Serial.println("ERROR:INVALID_DRIVE_FORMAT");
    }
  } else {
    Serial.println("ERROR:UNKNOWN_COMMAND");
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
    } else {
      if (serialCommandBuffer.length() < 64) {
        serialCommandBuffer += c;
      }
    }
  }
}

// Communication Failsafe Watchdog: automatically stops motors if no command received within timeout
void checkFailsafe() {
  if (serialControlActive) {
    if (millis() - lastCommandTime > COMMAND_TIMEOUT) {
      execute_stop();
      serialControlActive = false;
    }
  }
}

void loop() {
  // 1. Process USB Serial commands from Raspberry Pi 5
  checkSerialInput();
  checkFailsafe();

  // 2. Autonomous sensor & state machine logic (only active when DPDT switch is ON and not overridden by serial)
  if (!serialControlActive) {
    DPDT_STATE = digitalRead(DPDT_Push_Button_Pin);

    if (DPDT_STATE == 1) { 
      US_Values(f_us, f1_us, f2_us, b_us, l_us, r_us);

      if (LOGIC_LOCK == 1) { 
        side_us_logic_fun();
      }

      if (line_count >= line_chk_count) {
        end_stop();
        bot_shutdown();
        LOGIC_LOCK = 0;
        line_count = 0; 
      }
    } else {
      // Keep bot in completely quiet stopped state when DPDT switch is off
      bot_shutdown();
    }
  }
}

void side_us_logic_fun() {              
  if (f_us > 0 && f_us < fus_slow_dist) {
    motor_forward(fus_slow_speed);
  } else {
    motor_forward(normal_speed);
  } 

  // Left wall obstacle avoidance: steer right
  if ((l_us < 30) && (l_us > 0)) { 
    rgb_led(255, 0, 50); 
    moveServoTo(servo_center + 10);

    if (rt_st_count == 0 && lt_st_count == 0) {
      lt_st_count = 1;
    }
  }
  // Centered between walls: keep straight
  else if ((l_us >= 30) && (r_us >= 30) && (l_us > 0) && (r_us > 0)) { 
    rgb_led(0, 255, 0);
    moveServoTo(servo_center);
  } 
  // Right wall obstacle avoidance: steer left
  else if ((r_us < 30) && (r_us > 0)) {
    rgb_led(255, 0, 50); 
    moveServoTo(servo_center - 10);

    if (lt_st_count == 0 && rt_st_count == 0) {
      rt_st_count = 1;
    }
  } else {
    // Default straight if no side walls detected
    moveServoTo(servo_center);
  }
}

void bot_shutdown() {
  motor_stop();
  moveServoTo(servo_center);
  rgb_led(0, 0, 0);
}

void left_stop() {
  rgb_led(255, 255, 255);
  if (left_right_arc_turn) {
    motor_forward(210);
    delay(1000);
  }
  moveServoTo(left_turn_angle);
  delay(1500);
  moveServoTo(right_turn_angle);
  delay(1500);
  moveServoTo(servo_center);
  delay(1000);
  motor_stop();
  rgb_led(0, 0, 0);
}

void right_stop() {
  rgb_led(255, 255, 255);
  if (left_right_arc_turn) {
    motor_forward(210);
    delay(1000);
  }
  moveServoTo(right_turn_angle);
  delay(1500);
  moveServoTo(left_turn_angle);
  delay(1500);
  moveServoTo(servo_center);
  delay(1000);
  motor_stop();
  rgb_led(0, 0, 0);
}

void end_stop() {
  if (lt_st_count == 1) {
    left_stop();
  } else if (rt_st_count == 1) {
    right_stop();
  } else {
    bot_shutdown();
  }
}