/*
 * crane_controller.ino
 * =====================
 * Arduino firmware for the gantry crane motor controller.
 *
 * Receives CSV commands from the vision PC over Serial:
 *   "x,y,z\n"   →  float metres, e.g. "1.2340,0.7850,0.0250\n"
 *
 * Responds with:
 *   "ACK\n"   — command accepted and motion started
 *   "BUSY\n"  — currently executing a move
 *   "ERR\n"   — parse error
 *
 * Motor driver assumed: stepper motors via STEP/DIR pins (e.g. A4988/DRV8825).
 * Adapt STEPS_PER_METRE and pin assignments to your hardware.
 *
 * Axes:
 *   X axis  — bridge travel (along runway rails)
 *   Y axis  — trolley travel (along bridge)
 *   Z axis  — hoist (up/down)
 */

#include <AccelStepper.h>

// ── Pin definitions ────────────────────────────────────────────────────────
#define X_STEP_PIN   2
#define X_DIR_PIN    3
#define Y_STEP_PIN   4
#define Y_DIR_PIN    5
#define Z_STEP_PIN   6
#define Z_DIR_PIN    7
#define ENABLE_PIN   8    // LOW = motors enabled (common enable for all drivers)

// ── Motion parameters ─────────────────────────────────────────────────────
// Adjust these to match your mechanical setup:
//   steps_per_rev × microsteps / (lead_screw_pitch_m  OR  pulley_circumference_m)
const float STEPS_PER_METRE_X = 4000.0f;
const float STEPS_PER_METRE_Y = 4000.0f;
const float STEPS_PER_METRE_Z = 8000.0f;

const float MAX_SPEED_STEPS    = 3000.0f;   // steps/s
const float ACCELERATION_STEPS = 1500.0f;   // steps/s²

// ── Stepper objects ────────────────────────────────────────────────────────
AccelStepper stepperX(AccelStepper::DRIVER, X_STEP_PIN, X_DIR_PIN);
AccelStepper stepperY(AccelStepper::DRIVER, Y_STEP_PIN, Y_DIR_PIN);
AccelStepper stepperZ(AccelStepper::DRIVER, Z_STEP_PIN, Z_DIR_PIN);

// ── State ──────────────────────────────────────────────────────────────────
bool   busy         = false;
float  current_x    = 0.0f;   // metres (tracked from commands)
float  current_y    = 0.0f;
float  current_z    = 0.0f;
String inputBuffer  = "";

// ──────────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);

    pinMode(ENABLE_PIN, OUTPUT);
    digitalWrite(ENABLE_PIN, LOW);   // enable drivers

    stepperX.setMaxSpeed(MAX_SPEED_STEPS);
    stepperX.setAcceleration(ACCELERATION_STEPS);

    stepperY.setMaxSpeed(MAX_SPEED_STEPS);
    stepperY.setAcceleration(ACCELERATION_STEPS);

    stepperZ.setMaxSpeed(MAX_SPEED_STEPS);
    stepperZ.setAcceleration(ACCELERATION_STEPS);

    // Set current position as zero
    stepperX.setCurrentPosition(0);
    stepperY.setCurrentPosition(0);
    stepperZ.setCurrentPosition(0);

    Serial.println("CRANE_READY");
}

// ──────────────────────────────────────────────────────────────────────────
void loop() {
    // ── Read incoming serial line ─────────────────────────────────────────
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\n') {
            processCommand(inputBuffer);
            inputBuffer = "";
        } else {
            inputBuffer += c;
        }
    }

    // ── Run steppers ──────────────────────────────────────────────────────
    stepperX.run();
    stepperY.run();
    stepperZ.run();

    // Detect motion completion
    if (busy &&
        !stepperX.isRunning() &&
        !stepperY.isRunning() &&
        !stepperZ.isRunning())
    {
        busy = false;
        Serial.println("DONE");   // optional completion notification
    }
}

// ──────────────────────────────────────────────────────────────────────────
void processCommand(String cmd) {
    cmd.trim();
    if (cmd.length() == 0) return;

    if (busy) {
        Serial.println("BUSY");
        return;
    }

    // Parse "x,y,z"
    int comma1 = cmd.indexOf(',');
    int comma2 = cmd.lastIndexOf(',');

    if (comma1 == -1 || comma1 == comma2) {
        Serial.println("ERR");
        return;
    }

    float tx = cmd.substring(0, comma1).toFloat();
    float ty = cmd.substring(comma1 + 1, comma2).toFloat();
    float tz = cmd.substring(comma2 + 1).toFloat();

    // Convert metres to steps (absolute)
    long target_steps_x = (long)(tx * STEPS_PER_METRE_X);
    long target_steps_y = (long)(ty * STEPS_PER_METRE_Y);
    long target_steps_z = (long)(tz * STEPS_PER_METRE_Z);

    stepperX.moveTo(target_steps_x);
    stepperY.moveTo(target_steps_y);
    stepperZ.moveTo(target_steps_z);

    current_x = tx;
    current_y = ty;
    current_z = tz;
    busy = true;

    Serial.println("ACK");
}
