# TinyML Gesture Recognition for Sterile Medical Interface

A complete end-to-end implementation of gesture-based control for medical imaging viewers using TinyML and Edge Impulse, designed for sterile surgical environments where touchless interaction is critical.

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Implementation](#implementation)
- [Results](#results)
- [Hardware Deployment](#hardware-deployment)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project demonstrates a practical TinyML application for controlling medical imaging displays through hand gestures. The system processes 6-axis IMU (Inertial Measurement Unit) data to recognize seven distinct hand gestures, enabling surgeons to navigate MRI scans without breaking sterility by touching contaminated screens.

**Key Features:**
- Real-time gesture recognition at 50Hz sampling rate
- Runs on microcontrollers with only 8.5KB RAM and 52KB Flash
- 81.68% accuracy across 7 gesture classes
- 61ms inference latency enabling responsive control
- Complete workflow from data preprocessing to embedded deployment

## Problem Statement

During surgical procedures, surgeons need to reference medical imaging (MRI, CT scans) for guidance. However, maintaining a sterile field is paramount:

**Current Limitations:**
- Touching displays contaminates the sterile field
- Verbal commands to assistants create communication delays
- Foot pedals offer limited control options
- Voice control fails in noisy operating room environments

**Our Solution:**
A disposable smart glove with embedded IMU sensors that recognizes hand gestures, enabling direct control of imaging displays without physical contact or verbal communication.

## System Architecture
┌─────────────────────────────────────────────────────────────┐
│ End-to-End Pipeline │
├─────────────────────────────────────────────────────────────┤
│ │
│ Raw Sensor Data (Kaggle) │
│ ↓ │
│ Data Preprocessing & Visualization │
│ ↓ │
│ Edge Impulse Training │
│ - Spectral Feature Extraction (FFT) │
│ - 1D CNN Neural Network │
│ - Int8 Quantization │
│ ↓ │
│ TFLite Model (20KB) │
│ ↓ │
│ Deployment Options: │
│ - Python MRI Viewer Simulation │
│ - Wokwi Hardware Simulation │
│ - Physical ESP32/Arduino │
│ │
└─────────────────────────────────────────────────────────────┘


## Dataset

**Source:** [IMU Glove Dataset](https://www.kaggle.com/datasets/harrisonlou/imu-glove) from Kaggle

**Description:**
- Three MPU6050 IMU sensors placed on a glove (palm, index finger, thumb)
- 100 samples per gesture at 50Hz (2-second windows)
- 18 features per timestep (6 axes × 3 IMUs)
- Converted from ROS bag format to clean CSV files

**Gesture Classes:**
1. STATIC - No movement
2. SLIDE_UP - Upward swipe motion
3. SLIDE_DOWN - Downward swipe motion
4. SLIDE_LEFT - Left swipe motion
5. SLIDE_RIGHT - Right swipe motion
6. GRASP - Closing hand/gripping motion
7. RELEASE - Opening hand motion

**Data Preprocessing:**
- Extracted 18-axis IMU data (acceleration + gyroscope for 3 sensors)
- Normalized to 100 samples per gesture
- Stratified 80/20 train-test split
- Balanced class distribution verified

## Implementation

### Step 1: Data Engineering & Visualization

**Scripts:**
- `data_preparation.py` - Loads and organizes raw sensor data
- `visualizations.py` - Creates presentation-ready plots

**Outputs:**
- Signal comparison plots showing distinct gesture patterns
- Class distribution bar chart proving balanced dataset
- Multi-IMU sensor fusion visualization

**Key Insight:** Different gestures produce unique frequency signatures in the IMU data, making them distinguishable by machine learning models.

### Step 2: Model Training in Edge Impulse

**Platform:** [Edge Impulse Studio](https://studio.edgeimpulse.com/)

**Pipeline Configuration:**
- **Input Window:** 2000ms at 50Hz (100 samples)
- **Processing Block:** Spectral Analysis using FFT
  - Converts time-domain signals to frequency-domain features
  - Reduces 1800 raw values to 234 spectral features
  - Applies Hamming window for noise reduction
- **Learning Block:** 1D Convolutional Neural Network
  - Conv1D layers with 8 and 16 filters
  - MaxPooling and Dropout for regularization
  - Dense layer with 32 neurons
  - Softmax output for 7-8 classes

**Training Results:**
- Validation Accuracy: 81.68%
- Training completed in 50 epochs
- Confusion matrix shows good class separation
- Feature explorer demonstrates distinct gesture clusters

See `captures/` directory for Edge Impulse screenshots including model performance metrics, confusion matrix, and feature visualization.

### Step 3: Model Optimization for TinyML

**Quantization:**
- Converted from Float32 to Int8 representation
- 75% reduction in model size (207KB → 52KB)
- Minimal accuracy loss (~0.8%)

**Performance Metrics:**

| Metric | Float32 (Estimated) | Int8 (Quantized) | Improvement |
|--------|---------------------|------------------|-------------|
| Flash Usage | 207 KB | 51.8 KB | 75% reduction |
| RAM Usage | 34 KB | 8.5 KB | 75% reduction |
| Inference Time | 79 ms | 61 ms | 23% faster |
| Accuracy | 82.5% | 81.68% | -0.8% loss |

**Deployment Package:**
- TensorFlow Lite model: `tflite_learn_881836_3.tflite` (20.16 KB)
- Edge Impulse SDK with preprocessing pipeline
- Ready for Arduino, ESP32, and ARM Cortex-M deployment

### Step 4: MRI Viewer Application

**Interactive Simulation:** `mri_gesture_viewer.py`

A complete GUI application demonstrating the practical use case:

**Features:**
- Displays realistic brain MRI stack (30 slices)
- Gesture-controlled navigation through medical images
- Real-time gesture recognition with confidence display
- Smooth transformations (zoom, rotate, slice navigation)

**Gesture Mappings:**
- SLIDE_RIGHT → Next MRI slice
- SLIDE_LEFT → Previous MRI slice  
- SLIDE_UP → Zoom in
- SLIDE_DOWN → Zoom out
- GRASP → Rotate view counterclockwise
- RELEASE → Rotate view clockwise

**MRI Image Generation:**
The `generate_enhanced_mri.py` script creates clinical-grade synthetic brain scans with:
- Realistic anatomical structures (skull, gray matter, white matter, ventricles)
- Simulated pathology (tumor with peritumoral edema in middle slices)
- DICOM-style medical annotations
- 512x512 high-resolution images suitable for zoom operations

### Step 5: Hardware Deployment

**Wokwi Simulation:** Complete ESP32 + IMU circuit simulation available at `wokwi_simulation/`

**Hardware Components:**
- ESP32 DevKit (520KB RAM, 4MB Flash)
- MPU6050 6-axis IMU sensor
- RGB LEDs for visual feedback
- I2C communication protocol

**Arduino Implementation Highlights:**
- Non-blocking 50Hz sampling loop
- Circular buffer for 100-sample window
- I2C sensor reading via Adafruit library
- LED color coding for gesture indication

**Power Consumption:**
- Active inference: 80mA at 3.3V
- Estimated battery life: 12+ hours continuous operation (1000mAh LiPo)
- Sleep mode optimization enables multi-day operation

See `wokwi_simulation/circuit_screenshot.png` for hardware setup visualization.

## Results

### Model Performance

**Validation Accuracy:** 81.68%

**Practical Performance:**
- Inference latency: 61ms (well below 100ms human perception threshold)
- Real-time operation at 16 inferences per second
- Confidence thresholding at 65% reduces false positives

**Resource Efficiency:**
- Fits on Arduino Nano 33 BLE (256KB RAM, 1MB Flash)
- Uses only 3.3% of available RAM
- Leaves ample space for application code

### Clinical Relevance

**Advantages for Surgical Use:**
- Maintains sterile field (no physical contact)
- Reduces communication overhead (direct control vs. verbal commands)
- Low latency enables natural interaction
- Disposable glove approach (single-use, cost-effective)
- Works in noisy OR environments (unlike voice control)

## Hardware Deployment

### Circuit Connections
MPU6050 IMU:
VCC → ESP32 3.3V
GND → ESP32 GND
SCL → ESP32 GPIO22 (I2C Clock)
SDA → ESP32 GPIO21 (I2C Data)

LED Indicators:
Red LED → 220Ω → GPIO25 → GND
Green LED → 220Ω → GPIO26 → GND
Blue LED → 220Ω → GPIO27 → GND

### Deployment Steps

1. Install Arduino IDE and ESP32 board support
2. Install required libraries:
   - Adafruit MPU6050
   - Adafruit Unified Sensor
   - (Optional: TensorFlowLite_ESP32 for full model)
3. Upload sketch from `wokwi_simulation/sketch.ino`
4. Open Serial Monitor at 115200 baud to view predictions

## Project Structure
tinyml-gesture-recognition/
├── captures/ # Edge Impulse performance screenshots
│ ├── confusion_matrix.png
│ ├── feature_explorer.png
│ ├── model_testing.png
│ └── on_device_performance.png
│
├── data_preparation.py # Load and preprocess Kaggle dataset
├── visualizations.py # Generate signal plots and distributions
│
├── model_quantization.py # TFLite conversion and optimization
├── optimization_visualizations.py # Create comparison charts
│
├── generate_enhanced_mri.py # Synthetic medical image generation
├── mri_gesture_viewer.py # Interactive GUI application
│
├── wokwi_simulation/
│ ├── sketch.ino # Arduino code for ESP32
│ ├── diagram.json # Wokwi circuit configuration
│ ├── circuit_screenshot.png # Visual reference
│ └── README.md # Hardware-specific documentation
│
├── tflite_models/
│ └── tflite_learn_881836_3.tflite # Quantized model (20KB)
│
├── presentation_figures/ # Generated visualizations
│ ├── 1_signal_comparison.png
│ ├── 2_class_distribution.png
│ ├── 3_multi_imu_fusion.png
│ ├── 4_optimization_comparison.png
│ └── 5_memory_footprint.png
│
├── processed_data/
│ └── complete_dataset.csv # Cleaned and organized sensor data
│
├── mri_images/ # Synthetic brain scan stack
│ ├── brain_slice_000.png
│ ├── brain_slice_001.png
│ └── ... (30 slices total)
│
├── requirements.txt # Python dependencies
└── README.md # This file


## Getting Started

### Prerequisites

```bash
Python 3.8+
pip install -r requirements.txt
```

Quick Start   
1. Data Preparation:   

bash   
python data_preparation.py   
python visualizations.py   
2. Generate MRI Images:   
  
bash   
python generate_enhanced_mri.py   
3. Run Interactive Demo:   
   
bash   
python mri_gesture_viewer.py   
4. Explore Hardware Simulation:    
Visit Wokwi and load the files from wokwi_simulation/    
   
Full Workflow   
For complete model training and deployment:   
   
Download the IMU Glove dataset   
Run data preparation scripts   
Upload processed CSV to Edge Impulse    
Configure spectral analysis and train 1D CNN   
Export quantized TFLite model    
Deploy to embedded hardware or run Python simulation    
  
