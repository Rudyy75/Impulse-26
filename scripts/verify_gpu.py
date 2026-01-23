"""
GPU Verification Script for EchoFind

Run this script to verify that your environment is properly configured.
It checks:
1. PyTorch installation
2. CUDA availability
3. GPU memory
4. torchaudio for audio processing
5. All required packages

Run: python scripts/verify_gpu.py
"""

import sys

def check_pytorch():
    """Check PyTorch installation and CUDA availability."""
    print("=" * 60)
    print("PYTORCH CHECK")
    print("=" * 60)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024 ** 3)
                print(f"\nGPU {i}: {props.name}")
                print(f"  - Memory: {memory_gb:.2f} GB")
                print(f"  - Compute Capability: {props.major}.{props.minor}")
            
            # Quick performance test
            print("\nRunning quick GPU test...")
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.matmul(x, x)
            torch.cuda.synchronize()
            print("GPU computation successful!")
            
            return True
        else:
            print("\nWARNING: CUDA not available. Training will be slow on CPU.")
            print("For hackathon, GPU is highly recommended.")
            return False
            
    except ImportError:
        print("ERROR: PyTorch not installed!")
        print("Install with: pip install torch torchaudio")
        return False

def check_torchaudio():
    """Check torchaudio installation."""
    print("\n" + "=" * 60)
    print("TORCHAUDIO CHECK")
    print("=" * 60)
    
    try:
        import torchaudio
        print(f"torchaudio version: {torchaudio.__version__}")
        
        # Check available backends
        print(f"Available backends: {torchaudio.list_audio_backends()}")
        
        return True
        
    except ImportError:
        print("ERROR: torchaudio not installed!")
        return False

def check_other_packages():
    """Check other required packages."""
    print("\n" + "=" * 60)
    print("PACKAGE CHECK")
    print("=" * 60)
    
    packages = [
        "librosa",
        "numpy",
        "sklearn",
        "faiss",  # Will fail if not installed, that's OK for now
        "matplotlib",
        "tqdm",
        "pandas",
    ]
    
    all_ok = True
    
    for pkg in packages:
        try:
            # Special case: sklearn is imported as sklearn but package is scikit-learn
            if pkg == "sklearn":
                import sklearn
                print(f"scikit-learn: {sklearn.__version__}")
            elif pkg == "faiss":
                try:
                    import faiss
                    print(f"faiss: installed")
                except ImportError:
                    print(f"faiss: NOT INSTALLED (install with: pip install faiss-cpu)")
                    # Don't fail for faiss - we can install it later
            else:
                mod = __import__(pkg)
                version = getattr(mod, '__version__', 'installed')
                print(f"{pkg}: {version}")
        except ImportError:
            print(f"{pkg}: NOT INSTALLED")
            all_ok = False
    
    return all_ok

def check_memory_estimate():
    """Estimate if we have enough memory for training."""
    print("\n" + "=" * 60)
    print("MEMORY ESTIMATE FOR TRAINING")
    print("=" * 60)
    
    # Our model: ResNet-18 ~ 11M parameters
    # Batch of 64 spectrograms: 64 x 1 x 128 x 216 x 4 bytes = 7MB
    # With gradients and optimizer states: roughly 4x model size
    
    model_params = 11e6
    model_size_mb = model_params * 4 / (1024 ** 2)  # float32
    batch_size_mb = 64 * 1 * 128 * 216 * 4 / (1024 ** 2)
    
    # Rough estimate: model + gradients + optimizer + batch + buffer
    estimated_usage_gb = (model_size_mb * 4 + batch_size_mb * 3) / 1024 + 0.5
    
    print(f"Model size (float32): {model_size_mb:.1f} MB")
    print(f"Batch size (64 spectrograms): {batch_size_mb:.1f} MB")
    print(f"Estimated GPU memory usage: {estimated_usage_gb:.1f} GB")
    print(f"Recommended minimum: 4 GB (8 GB preferred)")
    
    try:
        import torch
        if torch.cuda.is_available():
            available_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if available_gb >= 4:
                print(f"\nYour GPU ({available_gb:.1f} GB) should be sufficient!")
            else:
                print(f"\nWARNING: Your GPU ({available_gb:.1f} GB) may be too small.")
                print("Consider reducing batch size to 32 or using Colab.")
    except:
        pass

def main():
    """Run all checks."""
    print("\n" + "=" * 60)
    print("ECHOFIND ENVIRONMENT VERIFICATION")
    print("=" * 60 + "\n")
    
    pytorch_ok = check_pytorch()
    audio_ok = check_torchaudio()
    packages_ok = check_other_packages()
    check_memory_estimate()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if pytorch_ok and audio_ok:
        print("\nEnvironment is ready for EchoFind!")
        print("\nNext steps:")
        print("1. Download FMA-Small dataset: https://github.com/mdeff/fma")
        print("2. Extract to echofind/data/fma_small/")
        print("3. Proceed to Phase 1: Input Pipeline")
        return 0
    else:
        print("\nEnvironment setup incomplete. Please install missing packages:")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
