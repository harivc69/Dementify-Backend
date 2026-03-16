#!/usr/bin/env python3
"""
Enhanced training script for tabular-only baseline model with robust checkpoint management
"""

import pandas as pd
import torch
import argparse
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset_csv import CSVDataset
from adrd.model import ADRDModel

def parse_args():
    parser = argparse.ArgumentParser("Train tabular-only baseline model with enhanced checkpointing")
    
    # Data paths
    parser.add_argument('--train_path', default='data/processed/train.csv', type=str)
    parser.add_argument('--val_path', default='data/processed/val.csv', type=str)
    parser.add_argument('--test_path', default='data/processed/test.csv', type=str)
    parser.add_argument('--cnf_file', default='data/toml_files/custom_nacc_config_no_img.toml', type=str)
    
    # Model parameters
    parser.add_argument('--d_model', default=64, type=int)
    parser.add_argument('--nhead', default=4, type=int)
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--gamma', default=2.0, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    
    # Checkpoint options
    parser.add_argument('--ckpt_dir', default='checkpoints', type=str)
    parser.add_argument('--ckpt_interval', default=10, type=int, help='Save checkpoint every N epochs')
    parser.add_argument('--resume_from', default=None, type=str, help='Resume from checkpoint')
    
    # Device
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', type=str)
    
    return parser.parse_args()

def save_training_info(ckpt_dir, args, start_time):
    """Save training configuration and metadata"""
    info = {
        'start_time': start_time,
        'config': vars(args),
        'device': str(args.device),
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    with open(os.path.join(ckpt_dir, 'training_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

def main():
    args = parse_args()
    
    print("=" * 80)
    print("TRAINING TABULAR-ONLY BASELINE MODEL (ENHANCED)")
    print("=" * 80)
    print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Print configuration
    print("\nConfiguration:")
    print("-" * 80)
    for arg, value in vars(args).items():
        print(f"  {arg:25s}: {value}")
    
    # Create checkpoint directory
    os.makedirs(args.ckpt_dir, exist_ok=True)
    print(f"\n✓ Checkpoint directory: {args.ckpt_dir}")
    
    # Save training info
    start_time = datetime.now().isoformat()
    save_training_info(args.ckpt_dir, args, start_time)
    
    # Load datasets
    print("\n" + "-" * 80)
    print("Loading Datasets")
    print("-" * 80)
    
    print(f"\nLoading training dataset...")
    dat_trn = CSVDataset(
        dat_file=args.train_path,
        cnf_file=args.cnf_file,
        mode=0,
        img_mode=-1,
        arch='NonImg',
        transforms=None,
        stripped=None
    )
    print(f"✓ Training: {len(dat_trn)} samples")
    
    print(f"\nLoading validation dataset...")
    dat_val = CSVDataset(
        dat_file=args.val_path,
        cnf_file=args.cnf_file,
        mode=1,
        img_mode=-1,
        arch='NonImg',
        transforms=None,
        stripped=None
    )
    print(f"✓ Validation: {len(dat_val)} samples")
    
    print(f"\nLoading test dataset...")
    dat_tst = CSVDataset(
        dat_file=args.test_path,
        cnf_file=args.cnf_file,
        mode=2,
        img_mode=-1,
        arch='NonImg',
        transforms=None,
        stripped=None
    )
    print(f"✓ Test: {len(dat_tst)} samples")
    
    # Get label fractions
    label_fractions = dat_trn.label_fractions if hasattr(dat_trn, 'label_fractions') else None
    
    if label_fractions:
        print("\nLabel distribution:")
        for label, fraction in label_fractions.items():
            print(f"  {label:10s}: {fraction:.4f}")
    
    # Initialize model
    print("\n" + "-" * 80)
    print("Initializing Model")
    print("-" * 80)
    
    # Set checkpoint path for intermediate saves
    ckpt_path = os.path.join(args.ckpt_dir, 'best_model.pt')
    
    model = ADRDModel(
        src_modalities=dat_trn.src_modalities,
        tgt_modalities=dat_trn.tgt_modalities,
        d_model=args.d_model,
        nhead=args.nhead,
        img_net='NonImg',
        device=args.device,
        cuda_devices=[0] if 'cuda' in args.device else [],
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        gamma=args.gamma,
        weight_decay=args.weight_decay,
        label_fractions=label_fractions,
        ckpt_path=ckpt_path,
        save_intermediate_ckpts=True,  # Enable intermediate checkpoint saving
        load_from_ckpt=args.resume_from is not None,
        wandb_=0
    )
    
    print(f"✓ Model initialized")
    print(f"  Device: {args.device}")
    print(f"  d_model: {args.d_model}")
    print(f"  nhead: {args.nhead}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Gamma (focal loss): {args.gamma}")
    
    # Resume from checkpoint if specified
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"\n✓ Resuming from checkpoint: {args.resume_from}")
        try:
            model = ADRDModel.from_ckpt(args.resume_from, device=args.device)
            print(f"✓ Model loaded successfully")
        except Exception as e:
            print(f"⚠ Failed to load checkpoint: {e}")
            print(f"  Starting training from scratch")
    
    # Train model
    print("\n" + "=" * 80)
    print(f"STARTING TRAINING - {args.num_epochs} EPOCHS")
    print("=" * 80)
    print(f"\nCheckpoints will be saved to: {args.ckpt_dir}")
    print(f"Best model will be saved as: {ckpt_path}")
    print(f"Intermediate checkpoints: Every {args.ckpt_interval} epochs")
    
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print("\n" + "-" * 80)
    
    training_start = time.time()
    
    try:
        model.fit(
            dat_trn.features,
            dat_val.features,
            dat_trn.labels,
            dat_val.labels,
            img_train_trans=None,
            img_vld_trans=None,
            img_mode=-1
        )
        
        training_time = time.time() - training_start
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Total training time: {training_time/60:.1f} minutes ({training_time/3600:.2f} hours)")
        print(f"Average time per epoch: {training_time/args.num_epochs:.1f} seconds")
        
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("TRAINING INTERRUPTED BY USER")
        print("=" * 80)
        training_time = time.time() - training_start
        print(f"Training time before interruption: {training_time/60:.1f} minutes")
        print(f"\nCheckpoints saved in: {args.ckpt_dir}")
        return 1
    
    except Exception as e:
        print("\n\n" + "=" * 80)
        print("TRAINING FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save final model
    print("\n" + "-" * 80)
    print("Saving Final Model")
    print("-" * 80)
    
    final_ckpt = os.path.join(args.ckpt_dir, f'final_model_epoch_{args.num_epochs}.pt')
    try:
        torch.save({
            'epoch': args.num_epochs,
            'model_state_dict': model.net_.state_dict(),
            'config': vars(args),
            'training_time': training_time,
        }, final_ckpt)
        print(f"✓ Final model saved to: {final_ckpt}")
    except Exception as e:
        print(f"✗ Failed to save final model: {e}")
    
    # Evaluate on test set
    print("\n" + "-" * 80)
    print("Evaluating on Test Set")
    print("-" * 80)
    
    try:
        test_metrics = model.evaluate(dat_tst.features, dat_tst.labels)
        
        print("\nTest Set Performance:")
        for metric, value in test_metrics.items():
            if isinstance(value, dict):
                print(f"\n{metric}:")
                for k, v in value.items():
                    print(f"  {k:15s}: {v:.4f}")
            else:
                print(f"  {metric:15s}: {value:.4f}")
        
        # Save test metrics
        metrics_file = os.path.join(args.ckpt_dir, 'test_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(test_metrics, f, indent=2, default=str)
        print(f"\n✓ Test metrics saved to: {metrics_file}")
        
    except Exception as e:
        print(f"⚠ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Save training summary
    summary = {
        'end_time': datetime.now().isoformat(),
        'total_training_time_minutes': training_time / 60,
        'epochs_completed': args.num_epochs,
        'final_checkpoint': final_ckpt,
        'best_checkpoint': ckpt_path,
    }
    
    with open(os.path.join(args.ckpt_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("ALL TASKS COMPLETED")
    print("=" * 80)
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Time: {training_time/60:.1f} minutes")
    print(f"\nCheckpoints saved in: {args.ckpt_dir}")
    print(f"  - Best model: best_model.pt")
    print(f"  - Final model: final_model_epoch_{args.num_epochs}.pt")
    print(f"  - Training info: training_info.json")
    print(f"  - Test metrics: test_metrics.json")
    print(f"  - Training summary: training_summary.json")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

