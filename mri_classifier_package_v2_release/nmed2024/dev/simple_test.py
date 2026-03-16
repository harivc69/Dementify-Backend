#!/usr/bin/env python3
"""Simple test to verify imports work"""

import sys
from pathlib import Path

print("=" * 80)
print("SIMPLE IMPORT TEST")
print("=" * 80)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n1. Testing basic imports...")
try:
    import pandas as pd
    print("✓ pandas")
except Exception as e:
    print(f"✗ pandas: {e}")

try:
    import torch
    print(f"✓ torch (version {torch.__version__})")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"✗ torch: {e}")

print("\n2. Testing dataset import...")
try:
    from data.dataset_csv import CSVDataset
    print("✓ CSVDataset imported successfully")
except Exception as e:
    print(f"✗ CSVDataset import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n3. Testing model import...")
try:
    from adrd.model import ADRDModel
    print("✓ ADRDModel imported successfully")
except Exception as e:
    print(f"✗ ADRDModel import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n4. Testing data loading...")
try:
    dat_trn = CSVDataset(
        dat_file='data/processed/train.csv',
        cnf_file='data/toml_files/custom_nacc_config_no_img.toml',
        mode=0,
        img_mode=-1,
        arch='NonImg',
        transforms=None,
        stripped=None
    )
    print(f"✓ Training dataset loaded: {len(dat_trn)} samples")
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("ALL TESTS PASSED ✓")
print("=" * 80)
print("\nReady to proceed with full training test!")

