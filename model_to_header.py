import os

MODEL_PATH = "./tflite-model/tflite_learn_881836_3.tflite"
OUTPUT_PATH = "./arduino_deployment/model.h"

os.makedirs("./arduino_deployment", exist_ok=True)

def convert_tflite_to_header(tflite_path, output_path):
    """
    Convert .tflite model to C++ header array
    """
    with open(tflite_path, 'rb') as f:
        model_data = f.read()
    
    hex_array = ', '.join([f'0x{b:02x}' for b in model_data])
    
    header_content = f"""// Auto-generated TFLite model for Edge Impulse
// Model: Medical Gesture Recognition
// Input: [1, 234] int8
// Output: [1, 8] int8

#ifndef MODEL_H
#define MODEL_H

const unsigned char model_tflite[] = {{
  {hex_array}
}};

const unsigned int model_tflite_len = {len(model_data)};

#endif  // MODEL_H
"""
    
    with open(output_path, 'w') as f:
        f.write(header_content)
    
    print(f"Model converted to C header")
    print(f"   Size: {len(model_data)} bytes ({len(model_data)/1024:.2f} KB)")
    print(f"   Saved to: {output_path}")

if __name__ == "__main__":
    print("=" * 60)
    print("Converting TFLite Model for Arduino")
    print("=" * 60)
    
    convert_tflite_to_header(MODEL_PATH, OUTPUT_PATH)
    
    print("\nNext Steps:")
    print("1. Copy model.h to your Arduino sketch folder")
    print("2. Use with TensorFlow Lite for Microcontrollers")
    print("=" * 60)
