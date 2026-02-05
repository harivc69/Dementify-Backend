#!/usr/bin/env python3
"""
Dementify Smart Dependency Installer
====================================

Installs PyTorch with CUDA support by default.
Falls back to CPU-only if CUDA installation fails.
"""

import sys
import subprocess
import os

def main():
    print("=" * 60)
    print("🚀 Dementify Smart Dependency Installer")
    print("=" * 60)

    # 1. Upgrade pip
    print("\n📦 Upgrading pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    # 2. Install PyTorch (CUDA-first strategy)
    # Note: Using CUDA 12.4 index for compatibility with Python 3.13
    torch_packages = ["torch", "torchvision"]
    cuda_index_url = "https://download.pytorch.org/whl/cu124"
    
    print("\n🔥 Attempting to install PyTorch with CUDA 12.4 support...")
    cuda_install_cmd = [
        sys.executable, "-m", "pip", "install", "--force-reinstall"
    ] + torch_packages + ["--index-url", cuda_index_url]
    
    print(f"   Executing: {' '.join(cuda_install_cmd)}")
    
    try:
        subprocess.check_call(cuda_install_cmd)
        print("✅ PyTorch with CUDA installed successfully!")
    except subprocess.CalledProcessError:
        print("\n⚠️ CUDA installation failed. Falling back to CPU-only...")
        cpu_install_cmd = [sys.executable, "-m", "pip", "install"] + torch_packages
        print(f"   Executing: {' '.join(cpu_install_cmd)}")
        subprocess.check_call(cpu_install_cmd)
        print("✅ PyTorch (CPU-only) installed successfully!")

    # 3. Verify CUDA availability
    print("\n🔍 Verifying PyTorch installation...")
    verify_cmd = [
        sys.executable, "-c",
        "import torch; print(f'   PyTorch Version: {torch.__version__}'); print(f'   CUDA Available: {torch.cuda.is_available()}')"
    ]
    subprocess.run(verify_cmd)

    # 4. Install remaining requirements
    print("\n📥 Installing other dependencies from requirements.txt...")
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(req_file):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
    else:
        print(f"❌ requirements.txt not found at {req_file}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ Installation Complete!")
    print("=" * 60)
    print("To start the server, run:")
    print("   uvicorn app.main:app --reload")

if __name__ == "__main__":
    main()
