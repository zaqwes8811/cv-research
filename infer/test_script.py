#!/usr/bin/env python3
import numpy as np
from hailo_platform import HEF, VDevice, HailoStreamInterface, ConfigureParams, InferVStreams, InputVStreamParams, OutputVStreamParams

# Load model
hef = HEF("yolov8n.hef")
input_info = hef.get_input_vstream_infos()[0]
output_info = hef.get_output_vstream_infos()[0]

print(f"Output name: {output_info.name}")
print(f"Output shape: {output_info.shape}")

# Create dummy input
dummy_input = np.zeros((1, 3, 640, 640), dtype=np.uint8)

# Run inference
with VDevice() as device:
    params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = device.configure(hef, params)[0]
    
    input_params = InputVStreamParams.make_from_network_group(network_group)
    output_params = OutputVStreamParams.make_from_network_group(network_group)
    
    with network_group.activate(), \
         InferVStreams(network_group, input_params, output_params) as infer_pipeline:
        
        results = infer_pipeline.infer({input_info.name: dummy_input})
        output = results[output_info.name]
        
        print(f"\nOutput type: {type(output)}")
        print(f"Output length: {len(output)}")
        
        # Recursively explore the structure
        def explore_structure(obj, depth=0):
            indent = "  " * depth
            if isinstance(obj, list):
                print(f"{indent}List with {len(obj)} elements:")
                for i, item in enumerate(obj[:3]):  # Show first 3 only
                    print(f"{indent}  [{i}]: ", end="")
                    explore_structure(item, depth + 1)
                if len(obj) > 3:
                    print(f"{indent}  ... and {len(obj)-3} more")
            elif isinstance(obj, np.ndarray):
                print(f"Array shape={obj.shape}, dtype={obj.dtype}, range=[{obj.min():.3f}, {obj.max():.3f}]")
                if obj.size < 20:  # Print small arrays completely
                    print(f"{indent}  Values: {obj.flatten()}")
            else:
                print(f"{type(obj)}: {obj}")
        
        explore_structure(output)