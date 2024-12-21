#include <Arduino.h>
#include <AFMotor.h>

// Define motors
AF_DCMotor motor1(1); // mouth
AF_DCMotor motor2(2); // body
AF_DCMotor motor3(3); // tail

// Motor states
struct MotorState {
    unsigned long startTime;
    int duration;  // in milliseconds
    bool isActive;
    int targetSpeed;
};

MotorState motorStates[3];

void setup() {
    Serial.begin(115200);
    
    // Initialize motors
    motor1.setSpeed(0);
    motor2.setSpeed(0);
    motor3.setSpeed(0);
    
    motor1.run(RELEASE);
    motor2.run(RELEASE);
    motor3.run(RELEASE);
    
    // Initialize states
    for(int i = 0; i < 3; i++) {
        motorStates[i] = {0, 0, false, 0};
    }
}

void processMotorCommand(char* command) {
    char* motorStr = strtok(command, ",");
    char* speedStr = strtok(NULL, ",");
    char* durationStr = strtok(NULL, ",");
    
    if (!motorStr || !speedStr) return;
    
    int motorNum = atoi(motorStr);
    int speed = atoi(speedStr);
    int duration = durationStr ? atoi(durationStr) : 0;  // Duration in ms
    
    // Constrain values
    motorNum = constrain(motorNum, 1, 3);
    speed = constrain(speed, 0, 255);
    
    // Update motor state
    int idx = motorNum - 1;
    motorStates[idx].startTime = millis();
    motorStates[idx].duration = duration;
    motorStates[idx].isActive = true;
    motorStates[idx].targetSpeed = speed;
    
    // Set initial motor state
    switch(motorNum) {
        case 1:
            motor1.setSpeed(speed);
            motor1.run(speed > 0 ? FORWARD : RELEASE);
            break;
        case 2:
            motor2.setSpeed(speed);
            motor2.run(speed > 0 ? FORWARD : RELEASE);
            break;
        case 3:
            motor3.setSpeed(speed);
            motor3.run(speed > 0 ? FORWARD : RELEASE);
            break;
    }
}

void updateMotors() {
    unsigned long currentTime = millis();
    
    for(int i = 0; i < 3; i++) {
        if (motorStates[i].isActive && motorStates[i].duration > 0) {
            if (currentTime - motorStates[i].startTime >= motorStates[i].duration) {
                // Stop the motor after duration
                switch(i + 1) {
                    case 1:
                        motor1.setSpeed(0);
                        motor1.run(RELEASE);
                        break;
                    case 2:
                        motor2.setSpeed(0);
                        motor2.run(RELEASE);
                        break;
                    case 3:
                        motor3.setSpeed(0);
                        motor3.run(RELEASE);
                        break;
                }
                motorStates[i].isActive = false;
            }
        }
    }
}

void loop() {
    static char buffer[32];
    static uint8_t bufIndex = 0;
    
    // Process incoming commands
    while (Serial.available() > 0) {
        char c = Serial.read();
        
        if (c == '\n') {
            buffer[bufIndex] = '\0';
            processMotorCommand(buffer);
            bufIndex = 0;
        }
        else if (bufIndex < sizeof(buffer) - 1) {
            buffer[bufIndex++] = c;
        }
    }
    
    // Update motor states
    updateMotors();
}
