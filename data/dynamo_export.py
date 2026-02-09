import torch
import torch.onnx

def export_with_dynamo(weights_path, output_path='yolov5s.onnx'):
    """Use PyTorch 2.0's dynamo export which respects opset"""
    
    # Load model
    model = torch.load(weights_path, map_location='cpu', weights_only=False)
    if isinstance(model, dict):
        model = model['model']
    model.eval()
    
    # Export with dynamo
    export_output = torch.onnx.dynamo_export(
        model,
        torch.randn(1, 3, 640, 640)
    )
    
    # Save with specific opset
    export_output.save(output_path, opset=12)
    
    print(f"✓ Exported to {output_path} with opset 12")

# Usage
#export_with_dynamo('yolov5s.pt', 'yolov5s_dynamo.onnx')


#!/usr/bin/env python3
import torch
import torch.onnx
import warnings

def export_yolov5(weights_path, output_path='yolov5s.onnx', img_size=640):
    """Export YOLOv5 using standard torch.onnx.export"""
    
    # Suppress warnings
    warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Load model
    print(f"Loading {weights_path}...")
    device = torch.device('cpu')
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    
    if isinstance(ckpt, dict):
        model = ckpt['model']
    else:
        model = ckpt
    
    model.eval()
    model.float()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    print(f"Exporting to {output_path}...")
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,  # Use 12 for Hailo compatibility
        do_constant_folding=True,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes={
            'images': {0: 'batch'},
            'output': {0: 'batch'}
        },
        verbose=False
    )
    
    print(f"✓ Successfully exported to {output_path}")
    
    # Verify
    try:
        import onnx
        model = onnx.load(output_path)
        print(f"✓ Opset version: {model.opset_import[0].version}")
        print(f"✓ Input shape: {model.graph.input[0].type.tensor_type.shape}")
        print(f"✓ Outputs: {len(model.graph.output)}")
    except:
        print("⚠ Could not verify ONNX file")
    
    return output_path

if __name__ == "__main__":
    # Export
    export_yolov5('yolov5s.pt', 'yolov5s_exported.onnx')
    
    # Also export with Hailo-specific settings
    print("\nExporting with Hailo-specific settings...")
    export_yolov5('yolov5s.pt', 'yolov5s_hailo.onnx')
