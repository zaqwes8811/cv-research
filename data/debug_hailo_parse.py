#!/usr/bin/env python3
import onnx
import onnxruntime as ort
import numpy as np

def debug_onnx_for_hailo(onnx_path):
    """Debug ONNX model structure for Hailo"""
    
    print("=" * 60)
    print("ONNX Model Analysis for Hailo")
    print("=" * 60)
    
    # Load model
    model = onnx.load(onnx_path)
    
    # 1. Basic info
    print(f"\n1. Model Info:")
    print(f"   IR Version: {model.ir_version}")
    print(f"   Opset: {model.opset_import[0].version}")
    print(f"   Producer: {model.producer_name}")
    
    # 2. Inputs
    print(f"\n2. Inputs:")
    for input in model.graph.input:
        shape = []
        for dim in input.type.tensor_type.shape.dim:
            if dim.dim_value:
                shape.append(dim.dim_value)
            else:
                shape.append(dim.dim_param)
        print(f"   {input.name}: shape={shape}")
    
    # 3. Outputs
    print(f"\n3. Outputs:")
    for output in model.graph.output:
        shape = []
        for dim in output.type.tensor_type.shape.dim:
            if dim.dim_value:
                shape.append(dim.dim_value)
            else:
                shape.append(dim.dim_param)
        print(f"   {output.name}: shape={shape}")
    
    # 4. Operators used
    print(f"\n4. Operators used:")
    ops = {}
    for node in model.graph.node:
        if node.op_type not in ops:
            ops[node.op_type] = 0
        ops[node.op_type] += 1
    
    for op, count in sorted(ops.items()):
        print(f"   {op:20} {count}")
    
    # 5. Check for unsupported ops
    unsupported_for_hailo = ['ScatterND', 'NonMaxSuppression', 'TopK']
    print(f"\n5. Checking for potentially unsupported ops:")
    for op in unsupported_for_hailo:
        if op in ops:
            print(f"   ⚠ {op} may need special handling")
    
    # 6. Test with ONNX Runtime
    print(f"\n6. ONNX Runtime test:")
    try:
        sess = ort.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        input_shape = sess.get_inputs()[0].shape
        
        # Create dummy input
        if input_shape[0] == 'batch' or input_shape[0] == 0:
            input_shape = (1, *input_shape[2:])
        
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        outputs = sess.run(None, {input_name: dummy_input})
        
        print(f"   ✓ Inference successful")
        for i, out in enumerate(outputs):
            print(f"   Output {i}: shape={out.shape}, dtype={out.dtype}")
            
    except Exception as e:
        print(f"   ✗ Inference failed: {e}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        debug_onnx_for_hailo(sys.argv[1])
    else:
        print("Usage: python debug_hailo_parse.py <model.onnx>")
