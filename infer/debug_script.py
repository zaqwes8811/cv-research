#!/usr/bin/env python3
import numpy as np
from hailo_platform import HEF, VDevice, HailoStreamInterface, ConfigureParams, InferVStreams, InputVStreamParams, OutputVStreamParams
import cv2

# Load model
hef = HEF("yolov8n.hef")
input_info = hef.get_input_vstream_infos()[0]
output_info = hef.get_output_vstream_infos()[0]

print(f"Model loaded:")
print(f"  Input shape: {input_info.shape}")
print(f"  Output shape: {output_info.shape}")
print(f"  Input name: {input_info.name}")
print(f"  Output name: {output_info.name}")

# Load and preprocess image
img = cv2.imread("test_image.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resized = cv2.resize(img_rgb, (640, 640))
input_tensor = np.transpose(resized, (2, 0, 1))
input_tensor = np.expand_dims(input_tensor, axis=0)
input_tensor = np.ascontiguousarray(input_tensor, dtype=np.uint8)

print(f"\nInput tensor shape: {input_tensor.shape}")
print(f"Input tensor dtype: {input_tensor.dtype}")

# Run inference
print("\nRunning inference...")
with VDevice() as device:
    params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = device.configure(hef, params)[0]
    
    input_params = InputVStreamParams.make_from_network_group(network_group)
    output_params = OutputVStreamParams.make_from_network_group(network_group)
    
    with network_group.activate(), \
         InferVStreams(network_group, input_params, output_params) as infer_pipeline:
        
        results = infer_pipeline.infer({input_info.name: input_tensor})
        output = results[output_info.name]

print("\n" + "="*60)
print("RAW OUTPUT ANALYSIS")
print("="*60)

# Analyze the output structure
print(f"\nOutput type: {type(output)}")
print(f"Output length: {len(output)}")

def analyze_structure(obj, depth=0, max_depth=3):
    """Recursively analyze the structure"""
    indent = "  " * depth
    
    if isinstance(obj, list):
        print(f"{indent}List[{len(obj)}]")
        if len(obj) > 0 and depth < max_depth:
            # Show first element's type
            first = obj[0]
            if isinstance(first, (list, np.ndarray)):
                print(f"{indent}  First element: ", end="")
                analyze_structure(first, depth + 1)
            else:
                print(f"{indent}  Contains: {type(first)}")
                
            # Show some statistics
            if all(isinstance(x, (int, float)) for x in obj[:10]):
                print(f"{indent}  Values: {obj[:10]}")
    
    elif isinstance(obj, np.ndarray):
        print(f"ndarray shape={obj.shape}, dtype={obj.dtype}, size={obj.size}")
        if obj.size > 0:
            print(f"{indent}  Range: [{obj.min():.3f}, {obj.max():.3f}]")
            print(f"{indent}  Mean: {obj.mean():.3f}")
            if obj.size < 20:
                print(f"{indent}  Values: {obj.flatten()}")
    
    else:
        print(f"{type(obj)}")

analyze_structure(output)

# Try to extract detections with very low threshold
print("\n" + "="*60)
print("ATTEMPTING TO EXTRACT DETECTIONS WITH VERY LOW THRESHOLD (0.01)")
print("="*60)

def extract_all_detections(output, threshold=0.01):
    """Try to extract any detection above a very low threshold"""
    detections = []
    
    if isinstance(output, list) and len(output) > 0:
        batch = output[0]  # First batch
        
        if isinstance(batch, list):
            for class_id, class_output in enumerate(batch):
                if isinstance(class_output, list) and len(class_output) > 0:
                    for det in class_output:
                        if isinstance(det, (list, np.ndarray)) and len(det) >= 5:
                            score = float(det[4]) if len(det) > 4 else 0
                            if score > threshold:
                                detections.append({
                                    'class_id': class_id,
                                    'score': score,
                                    'values': det[:5] if len(det) >= 5 else det
                                })
    
    return detections

detections = extract_all_detections(output, threshold=0.01)
print(f"Found {len(detections)} potential detections with score > 0.01")

if detections:
    print("\nFirst 10 detections:")
    for i, det in enumerate(detections[:10]):
        print(f"  {i+1}. Class {det['class_id']}: score={det['score']:.4f}, values={det['values']}")
    
    # Show class distribution
    from collections import Counter
    class_counts = Counter([d['class_id'] for d in detections])
    print("\nClass distribution:")
    for class_id, count in class_counts.most_common(10):
        print(f"  Class {class_id}: {count} detections")
else:
    print("No detections found even at threshold 0.01")
    
    # Try to find any non-zero values in the output
    print("\nSearching for any non-zero values...")
    
    def find_non_zero(obj, path=""):
        if isinstance(obj, list):
            for i, item in enumerate(obj):
                find_non_zero(item, f"{path}[{i}]")
        elif isinstance(obj, np.ndarray):
            if obj.size > 0:
                non_zero = np.count_nonzero(obj)
                if non_zero > 0:
                    print(f"{path}: {non_zero} non-zero values out of {obj.size}")
                    if obj.size < 100:
                        print(f"  Values: {obj}")
        elif isinstance(obj, (int, float)) and obj != 0:
            print(f"{path}: {obj}")
    
    find_non_zero(output)