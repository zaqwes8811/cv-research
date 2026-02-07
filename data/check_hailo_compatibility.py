# check_hailo_compatibility.py
import onnx
import onnxruntime as ort

def check_hailo_compatibility(onnx_path):
    """Check if ONNX model is compatible with Hailo-8"""
    
    print("=" * 60)
    print("Hailo-8 ONNX Compatibility Check")
    print("=" * 60)
    
    # Load model
    model = onnx.load(onnx_path)
    
    # Get opset version
    opset = model.opset_import[0].version
    print(f"ONNX Opset: {opset}")
    print(f"Recommended opset for Hailo: 11-13")
    
    if opset > 15:
        print("⚠ Warning: Opset >15 may have compatibility issues")
    
    # Check operators
    print(f"\nOperators used ({len(model.graph.node)} total):")
    
    ops = {}
    for node in model.graph.node:
        op_type = node.op_type
        ops[op_type] = ops.get(op_type, 0) + 1
    
    # Categorize operators
    supported = []
    partial = []
    unsupported = []
    
    # Define categories
    fully_supported = {
        'Conv', 'ConvTranspose', 'Relu', 'LeakyRelu', 'Sigmoid', 'Tanh',
        'Add', 'Sub', 'Mul', 'Div', 'BatchNormalization', 'MaxPool',
        'AveragePool', 'GlobalAveragePool', 'Concat', 'Reshape', 'Flatten',
        'Transpose', 'Gemm', 'MatMul', 'Clip', 'Pad', 'Slice', 'Split',
        'ReduceMean', 'ReduceSum', 'Softmax', 'Identity', 'Constant',
        'Resize', 'Exp', 'Log', 'Pow', 'Sqrt', 'Abs', 'Neg', 'Cast',
        'Shape', 'Expand', 'Tile', 'Gather'
    }
    
    partially_supported = {
        'LSTM', 'GRU', 'Attention', 'Upsample', 'NonMaxSuppression',
        'TopK', 'DepthToSpace', 'SpaceToDepth', 'InstanceNormalization',
        'LayerNormalization'
    }
    
    problematic = {
        'Loop', 'Scan', 'If', 'RandomNormal', 'RandomUniform',
        'Range', 'Sequence*', 'Optional*', 'Complex*'
    }
    
    for op, count in sorted(ops.items()):
        if op in fully_supported:
            supported.append((op, count))
        elif op in partially_supported:
            partial.append((op, count))
        elif any(p in op for p in problematic):
            unsupported.append((op, count))
        else:
            # Unknown operator
            partial.append((op, count))
    
    # Print results
    print("\n✅ Fully Supported:")
    for op, count in sorted(supported):
        print(f"  {op:25} ({count})")
    
    print("\n⚠ Partially Supported (may have limitations):")
    for op, count in sorted(partial):
        print(f"  {op:25} ({count})")
    
    print("\n❌ Not Supported (will fail):")
    for op, count in sorted(unsupported):
        print(f"  {op:25} ({count})")
    
    # Check for specific issues
    print(f"\nSpecific Checks:")
    
    # Check for dynamic shapes
    dynamic_inputs = 0
    for input in model.graph.input:
        for dim in input.type.tensor_type.shape.dim:
            if dim.dim_param:  # Dynamic dimension
                dynamic_inputs += 1
    
    if dynamic_inputs > 0:
        print(f"⚠ {dynamic_inputs} inputs have dynamic shapes")
        print("  Hailo prefers fixed shapes for optimal performance")
    
    # Check for unsupported data types
    supported_dtypes = {
        1: 'FLOAT',    # float32
        6: 'INT32',    # int32
        7: 'INT64',    # int64 (limited)
        9: 'BOOL',     # bool
        10: 'FLOAT16', # float16
        11: 'DOUBLE',  # double (converted to float32)
    }
    
    print(f"\nData Types:")
    for tensor in model.graph.initializer:
        dtype = tensor.data_type
        if dtype in supported_dtypes:
            print(f"  ✓ {tensor.name[:30]:30} {supported_dtypes[dtype]}")
        else:
            print(f"  ✗ {tensor.name[:30]:30} Unknown type {dtype}")
    
    # Overall compatibility score
    total_ops = sum(ops.values())
    supported_ops_count = sum(count for op, count in supported)
    compatibility = (supported_ops_count / total_ops * 100) if total_ops > 0 else 0
    
    print(f"\nCompatibility Score: {compatibility:.1f}%")
    
    if unsupported:
        print(f"\n❌ Model has {len(unsupported)} unsupported operators")
        return False
    elif partial:
        print(f"\n⚠ Model has {len(partial)} operators with limitations")
        return True
    else:
        print(f"\n✅ Model appears fully compatible")
        return True

if __name__ == "__main__":
    check_hailo_compatibility(
	"/workspace/yolov5_fixed/yolov5s_hailo.onnx"
	#"/workspace/yolov5_fixed/yolov5s_hailo_simplified.onnx"
	)
