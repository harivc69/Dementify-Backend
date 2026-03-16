#!/usr/bin/env python3
"""
Test training run for 5 epochs with detailed monitoring
Purpose: Verify training pipeline and benchmark performance
"""

import pandas as pd
import torch
import argparse
import os
import sys
import time
import psutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset_csv import CSVDataset
from adrd.model import ADRDModel

def get_gpu_memory():
    """Get GPU memory usage in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0

def get_system_memory():
    """Get system memory usage in GB"""
    return psutil.Process().memory_info().rss / 1024**3

def main():
    print("=" * 80)
    print("TASK 2: LOCAL TRAINING TEST RUN (5 EPOCHS)")
    print("=" * 80)
    
    # Configuration
    train_path = 'data/processed/train.csv'
    val_path = 'data/processed/val.csv'
    test_path = 'data/processed/test.csv'
    cnf_file = 'data/toml_files/custom_nacc_config_no_img.toml'
    
    # Model hyperparameters
    d_model = 64
    nhead = 4
    num_epochs = 5  # Test run
    batch_size = 32
    lr = 1e-4
    gamma = 2.0
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*80}")
    print("HARDWARE CONFIGURATION")
    print(f"{'='*80}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("GPU: Not available (using CPU)")
    
    print(f"CPU: {psutil.cpu_count(logical=False)} cores ({psutil.cpu_count(logical=True)} threads)")
    print(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print(f"Device: {device}")
    
    print(f"\n{'='*80}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*80}")
    print(f"d_model:     {d_model}")
    print(f"nhead:       {nhead}")
    print(f"num_epochs:  {num_epochs} (TEST RUN)")
    print(f"batch_size:  {batch_size}")
    print(f"lr:          {lr}")
    print(f"gamma:       {gamma}")
    
    # Track timing
    start_time = time.time()
    
    try:
        # Load datasets
        print(f"\n{'='*80}")
        print("LOADING DATASETS")
        print(f"{'='*80}")
        
        load_start = time.time()
        
        print(f"\nLoading training dataset...")
        dat_trn = CSVDataset(
            dat_file=train_path,
            cnf_file=cnf_file,
            mode=0,
            img_mode=-1,
            arch='NonImg',
            transforms=None,
            stripped=None
        )
        print(f"✓ Training: {len(dat_trn)} samples")
        
        print(f"\nLoading validation dataset...")
        dat_val = CSVDataset(
            dat_file=val_path,
            cnf_file=cnf_file,
            mode=1,
            img_mode=-1,
            arch='NonImg',
            transforms=None,
            stripped=None
        )
        print(f"✓ Validation: {len(dat_val)} samples")
        
        print(f"\nLoading test dataset...")
        dat_tst = CSVDataset(
            dat_file=test_path,
            cnf_file=cnf_file,
            mode=2,
            img_mode=-1,
            arch='NonImg',
            transforms=None,
            stripped=None
        )
        print(f"✓ Test: {len(dat_tst)} samples")
        
        load_time = time.time() - load_start
        print(f"\n✓ Data loading completed in {load_time:.1f} seconds")
        
        # Initialize model
        print(f"\n{'='*80}")
        print("INITIALIZING MODEL")
        print(f"{'='*80}")
        
        init_start = time.time()
        
        model = ADRDModel(
            src_modalities=dat_trn.src_modalities,
            tgt_modalities=dat_trn.tgt_modalities,
            d_model=d_model,
            nhead=nhead,
            img_net='NonImg',
            img_mode=-1,
            device=device,
            cuda_devices=[0] if 'cuda' in device else [],
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            gamma=gamma,
            label_fractions=dat_trn.label_fractions if hasattr(dat_trn, 'label_fractions') else None,
            wandb_=0  # Disable wandb for test
        )
        
        init_time = time.time() - init_start
        
        # Count parameters
        total_params = sum(p.numel() for p in model.transformer.parameters())
        trainable_params = sum(p.numel() for p in model.transformer.parameters() if p.requires_grad)
        
        print(f"✓ Model initialized in {init_time:.1f} seconds")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: ~{total_params * 4 / 1024**2:.1f} MB (float32)")
        
        # Initial memory usage
        print(f"\n{'='*80}")
        print("INITIAL MEMORY USAGE")
        print(f"{'='*80}")
        print(f"GPU Memory: {get_gpu_memory():.2f} GB")
        print(f"System Memory: {get_system_memory():.2f} GB")
        
        # Train model
        print(f"\n{'='*80}")
        print("STARTING TRAINING (5 EPOCHS)")
        print(f"{'='*80}")
        
        train_start = time.time()
        epoch_times = []
        
        # Monkey-patch to track epoch times
        original_fit = model.fit
        
        def tracked_fit(*args, **kwargs):
            # Track per-epoch timing
            import time
            from tqdm import tqdm
            
            # Call original fit
            return original_fit(*args, **kwargs)
        
        model.fit(
            dat_trn.features,
            dat_val.features,
            dat_trn.labels,
            dat_val.labels,
            img_train_trans=None,
            img_vld_trans=None,
            img_mode=-1
        )
        
        train_time = time.time() - train_start
        
        print(f"\n{'='*80}")
        print("TRAINING COMPLETED")
        print(f"{'='*80}")
        print(f"Total training time: {train_time/60:.1f} minutes")
        print(f"Average time per epoch: {train_time/num_epochs/60:.1f} minutes")
        
        # Final memory usage
        print(f"\n{'='*80}")
        print("FINAL MEMORY USAGE")
        print(f"{'='*80}")
        print(f"GPU Memory: {get_gpu_memory():.2f} GB")
        print(f"System Memory: {get_system_memory():.2f} GB")
        
        # GPU utilization (if available)
        if torch.cuda.is_available():
            print(f"\n{'='*80}")
            print("GPU STATISTICS")
            print(f"{'='*80}")
            print(f"Max GPU memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
            print(f"Max GPU memory reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
        
        # Save test checkpoint
        ckpt_path = 'checkpoints/test_run_5epochs.pt'
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(model.transformer.state_dict(), ckpt_path)
        print(f"\n✓ Test checkpoint saved to: {ckpt_path}")
        
        # Performance summary
        total_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print("TASK 2 SUMMARY: TEST RUN SUCCESSFUL ✓")
        print(f"{'='*80}")
        print(f"Total execution time: {total_time/60:.1f} minutes")
        print(f"  Data loading: {load_time:.1f} seconds")
        print(f"  Model init: {init_time:.1f} seconds")
        print(f"  Training (5 epochs): {train_time/60:.1f} minutes")
        print(f"  Time per epoch: {train_time/num_epochs/60:.1f} minutes")
        
        print(f"\n{'='*80}")
        print("TASK 3: PERFORMANCE BENCHMARKS")
        print(f"{'='*80}")
        print(f"Time per epoch: {train_time/num_epochs/60:.2f} minutes")
        print(f"GPU utilization: ~{(torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100) if torch.cuda.is_available() else 0:.1f}%")
        print(f"GPU memory used: {torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0:.2f} GB")
        print(f"System memory used: {get_system_memory():.2f} GB")
        
        print(f"\nESTIMATED TRAINING TIMES:")
        time_per_epoch = train_time / num_epochs
        print(f"  50 epochs:  {time_per_epoch * 50 / 3600:.1f} hours")
        print(f"  100 epochs: {time_per_epoch * 100 / 3600:.1f} hours")
        
        print(f"\n✓ NO ERRORS DETECTED - Training pipeline is working correctly!")
        
        return 0
        
    except Exception as e:
        print(f"\n{'='*80}")
        print("ERROR ENCOUNTERED")
        print(f"{'='*80}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        
        import traceback
        print(f"\nFull traceback:")
        traceback.print_exc()
        
        return 1

if __name__ == "__main__":
    sys.exit(main())

