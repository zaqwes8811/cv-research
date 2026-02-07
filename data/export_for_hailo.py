#!/usr/bin/env python3
import torch
import onnx
import onnxsim
import numpy as np
import sys
sys.path.append('/workspace/yolov5')

from models.yolo import Model
from utils.torch_utils import select_device

def export_yolov5_for_hailo(weights_path, img_size=640, opset=12):
    """Export YOLOv5 optimized for Hailo"""
    
    print(f"Exporting {weights_path} for Hailo...")
    
    # Load model
    device = select_device('cpu')
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model = ckpt['model'] if isinstance(ckpt, dict) else ckpt
    
    # Convert to ONNX
    model.eval()
    model.float()
    
    # Input
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    # Export
    onnx_path = weights_path.replace('.pt', '_hailo.onnx')
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=opset,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes={'images': {0: 'batch'}, 'output': {0: 'batch'}},
        do_constant_folding=True,
        verbose=False
    )
    
    print(f"✓ Exported to {onnx_path}")
    
    # Simplify
    print("Simplifying ONNX model...")
    model = onnx.load(onnx_path)
    model_simp, check = onnxsim.simplify(model)
    
    if check:
        simplified_path = onnx_path.replace('.onnx', '_simplified.onnx')
        onnx.save(model_simp, simplified_path)
        print(f"✓ Simplified model: {simplified_path}")
        return simplified_path
    
    return onnx_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='YOLOv5 weights')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset')
    
    args = parser.parse_args()
    export_yolov5_for_hailo(args.weights, args.img_size, args.opset)
