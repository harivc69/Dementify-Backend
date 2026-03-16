#!/usr/bin/env python3
"""
Training script for expanded 198-feature model
"""

print("DEBUG: Script starting...")
import argparse
print("DEBUG: argparse imported")
import torch
print("DEBUG: torch imported")
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
print("DEBUG: Basic imports completed")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
print("DEBUG: Path added")

from adrd.model import ADRDModel
print("DEBUG: ADRDModel imported")
from data.dataset_csv import CSVDataset
print("DEBUG: CSVDataset imported")

def parse_args():
    parser = argparse.ArgumentParser("Training with 198 features")
    
    # Data paths
    parser.add_argument('--train_path', default='data/processed_corrected/train.csv', type=str)
    parser.add_argument('--val_path', default='data/processed_corrected/val.csv', type=str)
    parser.add_argument('--test_path', default='data/processed_corrected/test.csv', type=str)
    parser.add_argument('--cnf_file', default='data/toml_files/custom_nacc_config_corrected.toml', type=str)
    
    # Model hyperparameters (same as overnight training)
    parser.add_argument('--d_model', default=128, type=int)
    parser.add_argument('--nhead', default=8, type=int)
    parser.add_argument('--num_encoder_layers', default=6, type=int)
    parser.add_argument('--num_decoder_layers', default=6, type=int)
    
    # Training hyperparameters
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--batch_size', default=16, type=int)  # Reduced from 32 to 16 for faster iterations
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    
    # Output
    parser.add_argument('--checkpoint_dir', default='checkpoints_corrected', type=str)
    parser.add_argument('--save_every', default=10, type=int)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = CSVDataset(args.train_path, args.cnf_file, mode=0)
    val_dataset = CSVDataset(args.val_path, args.cnf_file, mode=0)
    
    print(f"Train samples: {len(train_dataset.features)}")
    print(f"Val samples: {len(val_dataset.features)}")
    print(f"Features: {len(train_dataset.src_modalities)}")
    
    # Initialize model
    print("\nInitializing model...")
    model = ADRDModel(
        src_modalities=train_dataset.src_modalities,
        tgt_modalities=train_dataset.tgt_modalities,
        label_fractions=train_dataset.label_fractions,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        img_net='NonImg',  # No imaging data
        verbose=0,  # Disable verbose output to reduce I/O bottleneck
        _dataloader_num_workers=0  # FIX: Disable multiprocessing workers (WSL2 deadlock issue)
    )
    
    # Save training info
    training_info = {
        'start_time': datetime.now().isoformat(),
        'features': len(train_dataset.src_modalities),
        'train_samples': len(train_dataset.features),
        'val_samples': len(val_dataset.features),
        'config': {
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_encoder_layers': args.num_encoder_layers,
            'num_decoder_layers': args.num_decoder_layers,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay
        },
        'device': device,
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
    }
    
    with open(f'{args.checkpoint_dir}/training_info.json', 'w') as f:
        json.dump(training_info, f, indent=2)
    
    # Train model
    print("\n" + "="*80)
    print("STARTING TRAINING - 187 FEATURES (CORRECTED)")
    print("="*80)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Model dimension: {args.d_model}")
    print(f"Attention heads: {args.nhead}")
    print("="*80 + "\n")
    
    start_time = time.time()

    print("Starting model.fit()...")
    print(f"Time: {datetime.now().isoformat()}")
    sys.stdout.flush()

    # Fit model (signature: x_trn, x_vld, y_trn, y_vld)
    model.fit(
        train_dataset.features,
        val_dataset.features,
        train_dataset.labels,
        val_dataset.labels
    )

    training_time = time.time() - start_time
    print(f"\nTraining completed!")
    print(f"Time: {datetime.now().isoformat()}")
    sys.stdout.flush()
    
    # Save final model
    print("\nSaving final model...")
    final_checkpoint = {
        'epoch': args.epochs,
        'model_state_dict': model.net_.state_dict(),
        'config': training_info['config'],
        'training_time': training_time
    }
    
    torch.save(final_checkpoint, f'{args.checkpoint_dir}/final_model_epoch_{args.epochs}.pt')
    
    # Save training summary
    training_summary = {
        'end_time': datetime.now().isoformat(),
        'total_training_time_seconds': training_time,
        'total_training_time_minutes': training_time / 60,
        'epochs_completed': args.epochs,
        'average_time_per_epoch': training_time / args.epochs
    }
    
    with open(f'{args.checkpoint_dir}/training_summary.json', 'w') as f:
        json.dump(training_summary, f, indent=2)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Total time: {training_time/60:.1f} minutes")
    print(f"Average per epoch: {training_time/args.epochs:.1f} seconds")
    print(f"Final model saved to: {args.checkpoint_dir}/final_model_epoch_{args.epochs}.pt")
    print("="*80)

if __name__ == '__main__':
    main()

