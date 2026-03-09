#!/usr/bin/env python3
import cv2
import numpy as np
from hailo_platform import HEF, VDevice, HailoStreamInterface, ConfigureParams, InferVStreams, InputVStreamParams, OutputVStreamParams, FormatType

def preprocess_image(image_path, target_shape):
    """Simple image preprocessing"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_shape[2], target_shape[1]))  # width, height
    img = img.astype(np.float32) / 255.0  # Normalize
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    return img

def main():
    # Configuration
    hef_path = "resnet_v1_50.hef"
    image_path = "test_image.jpg"  # Your image
    
    # Load model
    hef = HEF(hef_path)
    input_info = hef.get_input_vstream_infos()[0]
    output_info = hef.get_output_vstream_infos()[0]
    
    print(f"Model loaded: {hef_path}")
    print(f"Input shape: {input_info.shape}")
    print(f"Output shape: {output_info.shape}")
    
    # Preprocess image
    image = preprocess_image(image_path, input_info.shape)
    image_batch = np.expand_dims(image, axis=0)
    
    # Run inference
    with VDevice() as device:
        params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = device.configure(hef, params)[0]
        
        input_params = InputVStreamParams.make_from_network_group(network_group)
        output_params = OutputVStreamParams.make_from_network_group(network_group)
        
        with network_group.activate(), \
             InferVStreams(network_group, input_params, output_params) as infer_pipeline:
            
            input_data = {input_info.name: image_batch}
            results = infer_pipeline.infer(input_data)
            output = results[output_info.name]
            
            print(f"Inference complete!")
            print(f"Output shape: {output.shape}")
            
            # For classification, get top prediction
            if len(output.shape) == 2:  # [batch, classes]
                top_class = np.argmax(output[0])
                confidence = output[0][top_class]
                print(f"Top class: {top_class}, confidence: {confidence:.3f}")

if __name__ == "__main__":
    main()