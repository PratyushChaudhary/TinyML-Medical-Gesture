import tensorflow as tf
import numpy as np

MODEL_PATH = "./tflite-model/tflite_learn_881836_3.tflite"

# Load model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("=" * 60)
print("Model Input/Output Analysis")
print("=" * 60)

print("\nINPUT DETAILS:")
for i, input_detail in enumerate(input_details):
    print(f"\nInput {i}:")
    print(f"  Name: {input_detail['name']}")
    print(f"  Shape: {input_detail['shape']}")
    print(f"  Type: {input_detail['dtype']}")
    print(f"  Quantization: {input_detail['quantization']}")

print("\nOUTPUT DETAILS:")
for i, output_detail in enumerate(output_details):
    print(f"\nOutput {i}:")
    print(f"  Name: {output_detail['name']}")
    print(f"  Shape: {output_detail['shape']}")
    print(f"  Type: {output_detail['dtype']}")
    print(f"  Quantization: {output_detail['quantization']}")

print("\n" + "=" * 60)
