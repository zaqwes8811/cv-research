import onnx

def verify_fixed_size(onnx_path):
    """Verify ONNX model has fixed input size"""
    
    model = onnx.load(onnx_path)
    graph = model.graph
    
    print(f"Model: {onnx_path}")
    print(f"Opset: {model.opset_import[0].version}")
    
    for i, input in enumerate(graph.input):
        print(f"\nInput {i}: {input.name}")
        shape = []
        dynamic_dims = []
        
        for j, dim in enumerate(input.type.tensor_type.shape.dim):
            if dim.dim_value:
                shape.append(dim.dim_value)
            elif dim.dim_param:
                shape.append(dim.dim_param)
                dynamic_dims.append(j)
        
        print(f"  Shape: {shape}")
        
        if dynamic_dims:
            print(f"  ⚠ Dynamic dimensions at indices: {dynamic_dims}")
            print(f"  Dimension names: {[input.type.tensor_type.shape.dim[i].dim_param for i in dynamic_dims]}")
        else:
            print("  ✓ All dimensions are fixed")
    
    return len(dynamic_dims) == 0

# Check your model
is_fixed = verify_fixed_size('/home/hailo/shared/shared_with_docker/yolov5s_fixed_640.onnx')
print(f"\nModel has fixed size: {is_fixed}")

