#include <Servo.h>

// Define pins
const int SPEEDO_PIN = 5;
const int STEERING_PIN = 6;
const int LEFT_LED_PIN = 4;
const int RIGHT_LED_PIN = 3;

// Servo objects
Servo speedoServo;
Servo steeringServo;

// Variables to store current states
int currentSpeed = 0;
int currentTurnSignal = 0;
int currentSteering = 0;

// Timing for turn signal flashing
unsigned long lastFlashTime = 0;
bool flashState = false;

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Attach servos to pins
  speedoServo.attach(SPEEDO_PIN);
  steeringServo.attach(STEERING_PIN);
  
  // Initialize LED pins as outputs
  pinMode(LEFT_LED_PIN, OUTPUT);
  pinMode(RIGHT_LED_PIN, OUTPUT);
  
  // Initialize all components to default positions
  setSpeed(0);
  setTurnSignal(0);
  setSteering(0);
  
  Serial.println("Dashboard ready");
}

void loop() {
  // Check for incoming serial data
  if (Serial.available() > 0) {
    String input = Serial.readString();
    input.trim(); // Remove whitespace
    
    // Parse the input string
    int firstSeparator = input.indexOf(';');
    int secondSeparator = input.indexOf(';', firstSeparator + 1);
    
    if (firstSeparator != -1 && secondSeparator != -1) {
      String speedStr = input.substring(0, firstSeparator);
      String turnStr = input.substring(firstSeparator + 1, secondSeparator);
      String steeringStr = input.substring(secondSeparator + 1);
      
      // Convert to integers
      int newSpeed = speedStr.toInt();
      int newTurn = turnStr.toInt();
      int newSteering = steeringStr.toInt();
      
      // Validate ranges
      if (newSpeed >= 0 && newSpeed <= 80) {
        setSpeed(newSpeed);
      }
      
      if (newTurn >= 0 && newTurn <= 3) {
        setTurnSignal(newTurn);
      }
      
      if (newSteering >= 0 && newSteering <= 2) {
        setSteering(newSteering);
      }
    }
  }
  
  // Handle turn signal flashing
  handleTurnSignal();
}

void setSpeed(int speed) {
  currentSpeed = speed;
  
  // Map speed to servo angle (0-180 degrees)
  // Assuming speedometer goes from 0 to 200 km/h
  int angle = map(speed, 0, 80, 180, 0);
  
  // Write to servo
  speedoServo.write(angle);
}

void setSteering(int steering) {
  currentSteering = steering;
  
  // Map steering to servo angle
  int angle;
  switch(steering) {
    case 0: // Forward
      angle = 135;
      break;
    case 1: // Left
      angle = 180;
      break;
    case 2: // Right
      angle = 45;
      break;
    default:
      angle = 135;
      break;
  }
  
  // Write to servo
  steeringServo.write(angle);
}

void setTurnSignal(int signal) {
  currentTurnSignal = signal;
  
  // Turn off all LEDs
  digitalWrite(LEFT_LED_PIN, LOW);
  digitalWrite(RIGHT_LED_PIN, LOW);
}

void handleTurnSignal() {
  unsigned long currentTime = millis();
  
  // Only update every 500ms for flashing
  if (currentTime - lastFlashTime > 500) {
    lastFlashTime = currentTime;
    flashState = !flashState;
    
    // Update LEDs based on current turn signal state
    switch(currentTurnSignal) {
      case 1: // Left flash
        digitalWrite(LEFT_LED_PIN, flashState ? HIGH : LOW);
        break;
      case 2: // Right flash
        digitalWrite(RIGHT_LED_PIN, flashState ? HIGH : LOW);
        break;
      case 3: // Hazards
        digitalWrite(LEFT_LED_PIN, flashState ? HIGH : LOW);
        digitalWrite(RIGHT_LED_PIN, flashState ? HIGH : LOW);
        break;
      default: // Off (0)
        // All LEDs already off
        break;
    }
  }
}
