#!/usr/bin/env python3
"""
Simple test to verify data loading works with CSVDataset
Run from nmed2024/dev directory
"""

import sys
import pandas as pd
import toml

print("=" * 80)
print("TESTING DATA LOADING - SIMPLE VERSION")
print("=" * 80)

# Test 1: Check files exist
print("\n" + "-" * 80)
print("Test 1: Check data files exist")
print("-" * 80)

train_file = 'data/processed/train.csv'
config_file = 'data/toml_files/custom_nacc_config_no_img.toml'

import os
print(f"Current directory: {os.getcwd()}")
print(f"Train file exists: {os.path.exists(train_file)}")
print(f"Config file exists: {os.path.exists(config_file)}")

if not os.path.exists(train_file):
    print(f"ERROR: {train_file} not found!")
    sys.exit(1)

if not os.path.exists(config_file):
    print(f"ERROR: {config_file} not found!")
    sys.exit(1)

# Test 2: Load CSV with pandas
print("\n" + "-" * 80)
print("Test 2: Load CSV with pandas")
print("-" * 80)

df = pd.read_csv(train_file)
print(f"✓ Loaded successfully")
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns[:10])}...")

# Test 3: Load TOML config
print("\n" + "-" * 80)
print("Test 3: Load TOML configuration")
print("-" * 80)

config = toml.load(config_file)
feature_names = list(config.get('feature', {}).keys())
label_names = list(config.get('label', {}).keys())

print(f"✓ Loaded successfully")
print(f"  Features: {len(feature_names)}")
print(f"  Labels: {len(label_names)}")
print(f"  First 5 features: {feature_names[:5]}")
print(f"  First 5 labels: {label_names[:5]}")

# Test 4: Import CSVDataset
print("\n" + "-" * 80)
print("Test 4: Import CSVDataset")
print("-" * 80)

try:
    from data.dataset_csv import CSVDataset
    print(f"✓ CSVDataset imported successfully")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Create dataset instance
print("\n" + "-" * 80)
print("Test 5: Create CSVDataset instance")
print("-" * 80)

try:
    dataset = CSVDataset(
        dat_file=train_file,
        cnf_file=config_file,
        mode=0,  # training mode
        img_mode=-1,  # no imaging
        arch='NonImg',
        transforms=None,
        stripped=None
    )
    print(f"✓ Dataset created successfully")
    print(f"  Length: {len(dataset)}")
    print(f"  Features shape: {dataset.features.shape if hasattr(dataset, 'features') else 'N/A'}")
    print(f"  Labels shape: {dataset.labels.shape if hasattr(dataset, 'labels') else 'N/A'}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Load a sample
print("\n" + "-" * 80)
print("Test 6: Load sample from dataset")
print("-" * 80)

try:
    sample = dataset[0]
    print(f"✓ Sample loaded successfully")
    print(f"  Type: {type(sample)}")
    
    if isinstance(sample, tuple):
        print(f"  Tuple length: {len(sample)}")
        for i, item in enumerate(sample):
            if hasattr(item, 'shape'):
                print(f"    Item {i}: shape={item.shape}, dtype={item.dtype}")
            elif isinstance(item, dict):
                print(f"    Item {i}: dict with keys={list(item.keys())}")
            else:
                print(f"    Item {i}: {type(item)}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Check label fractions
print("\n" + "-" * 80)
print("Test 7: Check label distribution")
print("-" * 80)

try:
    if hasattr(dataset, 'label_fractions'):
        print(f"✓ Label fractions available:")
        for label, fraction in dataset.label_fractions.items():
            print(f"    {label}: {fraction:.4f}")
    else:
        print(f"⚠ label_fractions attribute not found")
except Exception as e:
    print(f"⚠ Warning: {e}")

print("\n" + "=" * 80)
print("ALL TESTS PASSED ✓")
print("=" * 80)
print("\nData loading is working correctly!")
print("Ready to proceed with model training.")

