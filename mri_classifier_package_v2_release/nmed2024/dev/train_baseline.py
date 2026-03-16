#!/usr/bin/env python3
"""
Training script for tabular-only baseline model
Simplified version of train.py for our custom NACC dataset
"""

import pandas as pd
import torch
import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset_csv import CSVDataset
from adrd.model import ADRDModel
from tqdm import tqdm
import time

def parse_args():
    parser = argparse.ArgumentParser("Train tabular-only baseline model")
    
    # Data paths
    parser.add_argument('--train_path', default='data/processed/train.csv', type=str,
                        help='Path to training data CSV')
    parser.add_argument('--val_path', default='data/processed/val.csv', type=str,
                        help='Path to validation data CSV')
    parser.add_argument('--test_path', default='data/processed/test.csv', type=str,
                        help='Path to test data CSV')
    parser.add_argument('--cnf_file', default='data/toml_files/custom_nacc_config_no_img.toml', type=str,
                        help='Path to TOML configuration file')
    
    # Model parameters
    parser.add_argument('--d_model', default=64, type=int,
                        help='Dimension of feature embedding')
    parser.add_argument('--nhead', default=4, type=int,
                        help='Number of transformer heads')
    parser.add_argument('--num_epochs', default=50, type=int,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Learning rate')
    parser.add_argument('--gamma', default=2.0, type=float,
                        help='Gamma value for focal loss')
    parser.add_argument('--weight_decay', default=0.0, type=float,
                        help='Weight decay')
    
    # Training options
    parser.add_argument('--ckpt_path', default='checkpoints/baseline_model.pt', type=str,
                        help='Path to save model checkpoint')
    parser.add_argument('--load_from_ckpt', action='store_true',
                        help='Load model from checkpoint')
    parser.add_argument('--save_intermediate_ckpts', action='store_true',
                        help='Save intermediate checkpoints')
    parser.add_argument('--wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--balanced_sampling', action='store_true',
                        help='Use balanced sampling')
    
    # Device
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', type=str,
                        help='Device to use for training')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 80)
    print("TRAINING TABULAR-ONLY BASELINE MODEL")
    print("=" * 80)
    
    # Print configuration
    print("\nConfiguration:")
    print("-" * 80)
    for arg, value in vars(args).items():
        print(f"  {arg:25s}: {value}")
    
    # Create checkpoint directory
    ckpt_dir = os.path.dirname(args.ckpt_path)
    if ckpt_dir and not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        print(f"\n✓ Created checkpoint directory: {ckpt_dir}")
    
    # Initialize WandB if requested
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project="nmed2024-baseline",
                config=vars(args),
                name=f"baseline_d{args.d_model}_h{args.nhead}_lr{args.lr}"
            )
            print("\n✓ Weights & Biases initialized")
        except Exception as e:
            print(f"\n⚠ Failed to initialize WandB: {e}")
            args.wandb = False
    
    # Load datasets
    print("\n" + "-" * 80)
    print("Loading Datasets")
    print("-" * 80)
    
    print(f"\nLoading training dataset from {args.train_path}...")
    dat_trn = CSVDataset(
        dat_file=args.train_path,
        cnf_file=args.cnf_file,
        mode=0,  # training mode
        img_mode=-1,  # no imaging
        arch='NonImg',
        transforms=None,
        stripped=None
    )
    print(f"✓ Training dataset loaded: {len(dat_trn)} samples")
    
    print(f"\nLoading validation dataset from {args.val_path}...")
    dat_val = CSVDataset(
        dat_file=args.val_path,
        cnf_file=args.cnf_file,
        mode=1,  # validation mode
        img_mode=-1,  # no imaging
        arch='NonImg',
        transforms=None,
        stripped=None
    )
    print(f"✓ Validation dataset loaded: {len(dat_val)} samples")
    
    print(f"\nLoading test dataset from {args.test_path}...")
    dat_tst = CSVDataset(
        dat_file=args.test_path,
        cnf_file=args.cnf_file,
        mode=2,  # test mode
        img_mode=-1,  # no imaging
        arch='NonImg',
        transforms=None,
        stripped=None
    )
    print(f"✓ Test dataset loaded: {len(dat_tst)} samples")
    
    # Get label fractions for focal loss
    label_fractions = dat_trn.label_fractions if hasattr(dat_trn, 'label_fractions') else None
    
    if label_fractions:
        print("\nLabel distribution (training set):")
        for label, fraction in label_fractions.items():
            print(f"  {label:10s}: {fraction:.4f}")
    
    # Initialize model
    print("\n" + "-" * 80)
    print("Initializing Model")
    print("-" * 80)
    
    model = ADRDModel(
        src_modalities=dat_trn.src_modalities,
        tgt_modalities=dat_trn.tgt_modalities,
        d_model=args.d_model,
        nhead=args.nhead,
        img_net='NonImg',
        img_mode=-1,
        device=args.device,
        cuda_devices=[0] if 'cuda' in args.device else [],
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        gamma=args.gamma,
        weight_decay=args.weight_decay,
        label_fractions=label_fractions,
        balanced_sampling=args.balanced_sampling,
        wandb=args.wandb
    )
    
    print(f"✓ Model initialized")
    print(f"  Device: {args.device}")
    print(f"  d_model: {args.d_model}")
    print(f"  nhead: {args.nhead}")
    print(f"  Parameters: {sum(p.numel() for p in model.transformer.parameters()):,}")
    
    # Load from checkpoint if requested
    if args.load_from_ckpt and os.path.exists(args.ckpt_path):
        print(f"\nLoading model from checkpoint: {args.ckpt_path}")
        model = ADRDModel.from_ckpt(args.ckpt_path, device=args.device)
        print(f"✓ Model loaded from checkpoint")
    
    # Train model
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    
    start_time = time.time()
    
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
        
        training_time = time.time() - start_time
        print(f"\n✓ Training completed in {training_time/60:.1f} minutes")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        training_time = time.time() - start_time
        print(f"  Training time: {training_time/60:.1f} minutes")
    
    except Exception as e:
        print(f"\n\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save final model
    print("\n" + "-" * 80)
    print("Saving Model")
    print("-" * 80)
    
    try:
        torch.save(model.transformer.state_dict(), args.ckpt_path)
        print(f"✓ Model saved to: {args.ckpt_path}")
    except Exception as e:
        print(f"✗ Failed to save model: {e}")
    
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
        
        if args.wandb:
            wandb.log({"test_" + k: v for k, v in test_metrics.items() if not isinstance(v, dict)})
    
    except Exception as e:
        print(f"⚠ Evaluation failed: {e}")
    
    # Finish WandB
    if args.wandb:
        wandb.finish()
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

