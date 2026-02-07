# convert_opset18_to_12.py
import onnx
from onnx import version_converter
import onnxsim

def downgrade_opset(onnx_path, target_opset=12):
    """Downgrade ONNX opset from 18 to 12"""
    
    print(f"Converting {onnx_path} from opset 18 to {target_opset}...")
    
    # Load model
    model = onnx.load(onnx_path)
    current_opset = model.opset_import[0].version
    print(f"Current opset: {current_opset}")
    
    if current_opset <= target_opset:
        print(f"Model already opset {current_opset} <= {target_opset}")
        return onnx_path
    
    try:
        # Convert opset
        converted_model = version_converter.convert_version(model, target_opset)
        print(f"✓ Converted to opset {target_opset}")
        
        # Simplify
        model_simp, check = onnxsim.simplify(converted_model)
        if check:
            output_path = onnx_path.replace('.onnx', f'_opset{target_opset}.onnx')
            onnx.save(model_simp, output_path)
            print(f"✓ Saved to: {output_path}")
            return output_path
        else:
            output_path = onnx_path.replace('.onnx', f'_opset{target_opset}.onnx')
            onnx.save(converted_model, output_path)
            print(f"✓ Saved (not simplified): {output_path}")
            return output_path
            
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        downgrade_opset(sys.argv[1])
    else:
        print("Usage: python convert_opset18_to_12.py <model.onnx>")
