
# Try to load as PyTorch ZIP format
import torch

# Method 1: Standard load (should work for ZIP format)
try:
    model = torch.load('model.pt', map_location='cpu', weights_only=True)
    print("✓ Loaded as PyTorch ZIP format")
    print(f"Type: {type(model)}")
except Exception as e:
    print(f"✗ Not standard PyTorch ZIP: {e}")

# Method 2: Try with specific parameters
try:
    # PyTorch ZIP files often contain these files
    import zipfile
    zf = zipfile.ZipFile('model.pt')
    
    if 'archive/data.pkl' in zf.namelist() or 'data.pkl' in zf.namelist():
        print("✓ This is a PyTorch ZIP checkpoint")
        
        # Extract and load manually
        import io
        import pickle
        
        # Look for pickle file
        for name in zf.namelist():
            if name.endswith('.pkl') or 'data' in name:
                with zf.open(name) as f:
                    data = pickle.load(f)
                    print(f"Found: {name}, type: {type(data)}")
                    
except Exception as e:
    print(f"✗ Error examining ZIP: {e}")



# check_pytorch_compatibility.py
import torch
import struct

def check_pt_compatibility(filepath):
    """Check which PyTorch versions can load this file"""
    
    current_version = torch.__version__
    print(f"Current PyTorch: {current_version}")
    
    # Known PyTorch version ranges
    pytorch_versions = {
        '0.4': '2018',
        '1.0': '2018',
        '1.1': '2019', 
        '1.2': '2019',
        '1.3': '2019',
        '1.4': '2019',
        '1.5': '2020',
        '1.6': '2020',
        '1.7': '2020',
        '1.8': '2021',
        '1.9': '2021',
        '1.10': '2021',
        '1.11': '2021',
        '1.12': '2022',
        '1.13': '2022',
        '2.0': '2022',
        '2.1': '2023',
        '2.2': '2024',
        '2.3': '2024',
    }
    
    # Try different loading methods
    print("\nTrying to load with different methods:")
    
    # Method 1: weights_only=False (compatible with all)
    try:
        ckpt = torch.load(filepath, map_location='cpu', weights_only=False)
        print("✓ weights_only=False: SUCCESS")
    except Exception as e:
        print(f"✗ weights_only=False: FAILED - {str(e)[:50]}")
    
    # Method 2: weights_only=True (PyTorch 2.6+)
    try:
        ckpt = torch.load(filepath, map_location='cpu', weights_only=True)
        print("✓ weights_only=True: SUCCESS (PyTorch 2.6+ safe)")
    except Exception as e:
        print(f"✗ weights_only=True: FAILED - {str(e)[:50]}")
    
    # Check pickle protocol
    try:
        with open(filepath, 'rb') as f:
            # Read pickle protocol
            data = f.read(10)
            if data[0:2] == b'\x80\x04':  # Protocol 4
                print("\nPickle protocol: 4 (Python 3.4+)")
            elif data[0:2] == b'\x80\x03':  # Protocol 3
                print("\nPickle protocol: 3 (Python 3.0+)")
            else:
                print(f"\nPickle protocol: Unknown ({data[0:2].hex()})")
    except:
        pass
    
    print("\nCompatibility guide:")
    print("- PyTorch <1.6: May have issues with newer checkpoints")
    print("- PyTorch 1.6-1.13: Good compatibility")
    print("- PyTorch 2.0+: Best for recent models")
    print("- weights_only=True: Only in PyTorch 2.6+ for security")

if __name__ == "__main__":
    check_pt_compatibility("model.pt")
