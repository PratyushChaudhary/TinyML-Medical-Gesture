// ============================================
// TinyML Gesture Recognition - Wokwi Demo
// Hardware: ESP32 + MPU6050 IMU
// ============================================

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// For real deployment, include:
// #include <TensorFlowLite_ESP32.h>
// #include "model.h"

Adafruit_MPU6050 mpu;

// Configuration
#define SAMPLE_RATE_HZ 50
#define WINDOW_SIZE 100
#define NUM_AXES 6  // Ax, Ay, Az, Gx, Gy, Gz

// LED pins for gesture indication
#define LED_RED 25
#define LED_GREEN 26
#define LED_BLUE 27

// Data buffer
float sensorBuffer[WINDOW_SIZE][NUM_AXES];
int bufferIndex = 0;
bool bufferFull = false;

// Gesture labels
const char* gestures[] = {
  "GRASP", "RELEASE", "SLIDE_DOWN", "SLIDE_LEFT", 
  "SLIDE_RIGHT", "SLIDE_UP", "STATIC", "NONE"
};

// ============================================
// Setup
// ============================================
void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  
  Serial.println("================================");
  Serial.println("TinyML Gesture Recognition");
  Serial.println("Edge Impulse Deployment");
  Serial.println("================================");
  
  // Initialize LEDs
  pinMode(LED_RED, OUTPUT);
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_BLUE, OUTPUT);
  
  // LED startup sequence
  digitalWrite(LED_RED, HIGH);
  delay(200);
  digitalWrite(LED_RED, LOW);
  digitalWrite(LED_GREEN, HIGH);
  delay(200);
  digitalWrite(LED_GREEN, LOW);
  digitalWrite(LED_BLUE, HIGH);
  delay(200);
  digitalWrite(LED_BLUE, LOW);
  
  // Initialize IMU
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) {
      digitalWrite(LED_RED, !digitalRead(LED_RED));
      delay(100);
    }
  }
  
  Serial.println("MPU6050 Found!");
  
  // Configure IMU
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  
  Serial.println("\nConfiguration:");
  Serial.println("  Sample Rate: 50 Hz");
  Serial.println("  Window Size: 100 samples (2 seconds)");
  Serial.println("  Features: 6 axes (Ax, Ay, Az, Gx, Gy, Gz)");
  Serial.println("  Model: 234 spectral features → 8 gestures");
  Serial.println("\nStarting data collection...\n");
  
  digitalWrite(LED_GREEN, HIGH);
  delay(1000);
  digitalWrite(LED_GREEN, LOW);
}

// ============================================
// Main Loop
// ============================================
void loop() {
  static unsigned long lastSample = 0;
  unsigned long now = millis();
  
  // Sample at 50 Hz (every 20ms)
  if (now - lastSample >= 20) {
    lastSample = now;
    
    // Read IMU data
    sensors_event_t accel, gyro, temp;
    mpu.getEvent(&accel, &gyro, &temp);
    
    // Store in buffer
    sensorBuffer[bufferIndex][0] = accel.acceleration.x;
    sensorBuffer[bufferIndex][1] = accel.acceleration.y;
    sensorBuffer[bufferIndex][2] = accel.acceleration.z;
    sensorBuffer[bufferIndex][3] = gyro.gyro.x;
    sensorBuffer[bufferIndex][4] = gyro.gyro.y;
    sensorBuffer[bufferIndex][5] = gyro.gyro.z;
    
    bufferIndex++;
    
    // When buffer is full, run inference
    if (bufferIndex >= WINDOW_SIZE) {
      bufferIndex = 0;
      bufferFull = true;
      
      runInference();
    }
    
    // Visual feedback - blink during data collection
    if (bufferIndex % 25 == 0) {
      digitalWrite(LED_BLUE, !digitalRead(LED_BLUE));
    }
  }
}

// ============================================
// Inference (Simulated for Wokwi)
// ============================================
void runInference() {
  Serial.println("------------------------");
  Serial.println("Running Inference...");
  
  // In real deployment, this would:
  // 1. Extract spectral features using FFT
  // 2. Quantize to int8
  // 3. Run TFLite model
  // 4. Get prediction
  
  // For demo: simulate gesture detection
  int predictedClass = simulateGestureDetection();
  float confidence = 0.85 + random(0, 15) / 100.0;
  
  // Display result
  Serial.print("Gesture: ");
  Serial.print(gestures[predictedClass]);
  Serial.print(" (");
  Serial.print(confidence * 100, 1);
  Serial.println("%)");
  
  // Visual feedback
  indicateGesture(predictedClass);
  
  Serial.println("------------------------\n");
}

// ============================================
// Simulate Gesture Detection
// ============================================
int simulateGestureDetection() {
  // Analyze motion characteristics
  float accelMagnitude = 0;
  float gyroMagnitude = 0;
  
  for (int i = 0; i < WINDOW_SIZE; i++) {
    accelMagnitude += sqrt(
      sensorBuffer[i][0] * sensorBuffer[i][0] +
      sensorBuffer[i][1] * sensorBuffer[i][1] +
      sensorBuffer[i][2] * sensorBuffer[i][2]
    );
    
    gyroMagnitude += sqrt(
      sensorBuffer[i][3] * sensorBuffer[i][3] +
      sensorBuffer[i][4] * sensorBuffer[i][4] +
      sensorBuffer[i][5] * sensorBuffer[i][5]
    );
  }
  
  accelMagnitude /= WINDOW_SIZE;
  gyroMagnitude /= WINDOW_SIZE;
  
  // Simple heuristic for demo
  if (accelMagnitude < 10.0 && gyroMagnitude < 0.5) {
    return 6; // STATIC
  } else if (gyroMagnitude > 2.0) {
    return random(0, 2); // GRASP or RELEASE
  } else if (accelMagnitude > 12.0) {
    return random(2, 6); // SLIDE gestures
  }
  
  return 6; // Default to STATIC
}

// ============================================
// LED Indication
// ============================================
void indicateGesture(int gestureClass) {
  // Turn off all LEDs
  digitalWrite(LED_RED, LOW);
  digitalWrite(LED_GREEN, LOW);
  digitalWrite(LED_BLUE, LOW);
  
  // Light up based on gesture
  switch(gestureClass) {
    case 0: // GRASP
      digitalWrite(LED_GREEN, HIGH);
      break;
    case 1: // RELEASE
      digitalWrite(LED_BLUE, HIGH);
      break;
    case 2: // SLIDE_DOWN
      digitalWrite(LED_RED, HIGH);
      break;
    case 3: // SLIDE_LEFT
      digitalWrite(LED_RED, HIGH);
      digitalWrite(LED_BLUE, HIGH);
      break;
    case 4: // SLIDE_RIGHT
      digitalWrite(LED_GREEN, HIGH);
      digitalWrite(LED_BLUE, HIGH);
      break;
    case 5: // SLIDE_UP
      digitalWrite(LED_GREEN, HIGH);
      digitalWrite(LED_RED, HIGH);
      break;
    case 6: // STATIC
      // All off
      break;
    default:
      // Blink all
      for(int i = 0; i < 3; i++) {
        digitalWrite(LED_RED, HIGH);
        digitalWrite(LED_GREEN, HIGH);
        digitalWrite(LED_BLUE, HIGH);
        delay(100);
        digitalWrite(LED_RED, LOW);
        digitalWrite(LED_GREEN, LOW);
        digitalWrite(LED_BLUE, LOW);
        delay(100);
      }
  }
  
  delay(500);
}
