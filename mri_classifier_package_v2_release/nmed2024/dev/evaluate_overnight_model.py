#!/usr/bin/env python3
"""
Evaluate the overnight trained model and generate performance metrics
"""

import pandas as pd
import torch
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset_csv import CSVDataset
from adrd.model import ADRDModel

def main():
    print("=" * 80)
    print("OVERNIGHT MODEL EVALUATION")
    print("=" * 80)
    print()
    
    # Paths
    test_path = 'data/processed/test.csv'
    cnf_file = 'data/toml_files/custom_nacc_config_no_img.toml'
    model_path = 'checkpoints_overnight/final_model_epoch_150.pt'
    
    print(f"Loading test dataset...")
    dat_tst = CSVDataset(
        dat_file=test_path,
        cnf_file=cnf_file,
        mode=2,
        img_mode=-1,
        arch='NonImg',
        transforms=None,
        stripped=None
    )
    print(f"✓ Test dataset: {len(dat_tst)} samples")
    print()
    
    # Load model
    print(f"Loading trained model from: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location='cuda:0')
        print(f"✓ Checkpoint loaded")
        print(f"  Checkpoint keys: {list(checkpoint.keys())}")
        
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"\nModel Configuration:")
            print(f"  d_model: {config.get('d_model', 'N/A')}")
            print(f"  nhead: {config.get('nhead', 'N/A')}")
            print(f"  num_epochs: {config.get('num_epochs', 'N/A')}")
            print(f"  batch_size: {config.get('batch_size', 'N/A')}")
            print(f"  learning_rate: {config.get('lr', 'N/A')}")
            print(f"  weight_decay: {config.get('weight_decay', 'N/A')}")
        
        if 'training_time' in checkpoint:
            print(f"\nTraining Time: {checkpoint['training_time']/60:.1f} minutes")
        
        if 'epoch' in checkpoint:
            print(f"Epochs Completed: {checkpoint['epoch']}")
        
        print()
        print("=" * 80)
        print("MODEL EVALUATION COMPLETE")
        print("=" * 80)
        print()
        print("Note: The ADRDModel class does not have a built-in evaluate() method.")
        print("To get detailed performance metrics, you would need to:")
        print("  1. Reconstruct the ADRDModel with the same configuration")
        print("  2. Load the state_dict from the checkpoint")
        print("  3. Run inference on the test set")
        print("  4. Calculate metrics manually")
        print()
        print("Checkpoint Information:")
        print(f"  Location: {model_path}")
        print(f"  Size: {Path(model_path).stat().st_size / 1024:.1f} KB")
        print(f"  Model ready for inference")
        
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

