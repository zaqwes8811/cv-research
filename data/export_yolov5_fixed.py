import torch
import sys
sys.path.append('/workspace/yolov5')

from models.yolo import Model

def export_yolov5_fixed(weights_path, img_size=640, batch_size=1, opset=18):
    """Export YOLOv5 with fixed input size"""
    
    # Load model
    device = torch.device('cpu')
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model = ckpt['model'] if isinstance(ckpt, dict) else ckpt
    model.eval()
    model.float()
    
    # Fuse model (optional, for YOLOv5)
    if hasattr(model, 'fuse'):
        model.fuse()
    
    # Fixed dummy input
    dummy_input = torch.zeros(batch_size, 3, img_size, img_size, device=device)
    
    # Export
    output_path = weights_path.replace('.pt', f'_fixed_{img_size}.onnx')
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes=None,  # Fixed = no dynamic axes
        verbose=False
    )
    
    print(f"✓ Exported with fixed size {batch_size}x3x{img_size}x{img_size}")
    print(f"  Saved to: {output_path}")
    
    return output_path

# Usage
export_yolov5_fixed('yolov5s.pt', img_size=640, batch_size=1)
