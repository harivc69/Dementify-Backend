#!/usr/bin/env python3
"""
Evaluate the trained overnight model on all three splits
Uses the ADRDModel's built-in prediction capabilities
"""

import pandas as pd
import numpy as np
import torch
import sys
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, average_precision_score
)
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset_csv import CSVDataset
from adrd.model import ADRDModel

print("=" * 80)
print("MODEL EVALUATION - LOADING COMPONENTS")
print("=" * 80)

# Configuration
checkpoint_path = 'checkpoints_overnight/final_model_epoch_150.pt'
cnf_file = 'data/toml_files/custom_nacc_config_no_img.toml'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(f"\nDevice: {device}")
print(f"Checkpoint: {checkpoint_path}")

# Load checkpoint
print(f"\nLoading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location=device)
config = checkpoint['config']

print(f"✓ Checkpoint loaded")
print(f"  Epoch: {checkpoint['epoch']}")
print(f"  d_model: {config['d_model']}")
print(f"  nhead: {config['nhead']}")

# Load datasets
print(f"\n{'=' * 80}")
print("LOADING DATASETS")
print(f"{'=' * 80}")

datasets = {}
for split_name, data_path, mode in [
    ('train', 'data/processed/train.csv', 0),
    ('validation', 'data/processed/val.csv', 1),
    ('test', 'data/processed/test.csv', 2)
]:
    print(f"\nLoading {split_name} dataset...")
    dataset = CSVDataset(
        dat_file=data_path,
        cnf_file=cnf_file,
        mode=mode,
        img_mode=-1,
        arch='NonImg',
        transforms=None,
        stripped=None
    )
    print(f"✓ {split_name}: {len(dataset)} samples")
    datasets[split_name] = dataset

# Get label names
label_names = list(datasets['train'].tgt_modalities.keys())
print(f"\nLabel names ({len(label_names)}): {label_names}")

# Reconstruct and load model
print(f"\n{'=' * 80}")
print("RECONSTRUCTING MODEL")
print(f"{'=' * 80}")

print(f"\nInitializing ADRDModel...")
model = ADRDModel(
    src_modalities=datasets['train'].src_modalities,
    tgt_modalities=datasets['train'].tgt_modalities,
    label_fractions=datasets['train'].label_fractions,
    d_model=config['d_model'],
    nhead=config['nhead'],
    num_epochs=1,
    batch_size=config['batch_size'],
    lr=config['lr'],
    gamma=config['gamma'],
    weight_decay=config['weight_decay'],
    device=device,
    cuda_devices=[0] if 'cuda' in device else [],
    img_net='NonImg',
    verbose=0
)

print(f"✓ Model initialized")

# Initialize network
print(f"\nInitializing network...")
model._init_net()
print(f"✓ Network initialized")

# Load trained weights
print(f"\nLoading trained weights...")
model.net_.load_state_dict(checkpoint['model_state_dict'])
model.net_.eval()
print(f"✓ Weights loaded, model in eval mode")

# Function to get predictions
def evaluate_split(model, dataset, split_name):
    """Evaluate model on a dataset split"""
    print(f"\n{'=' * 80}")
    print(f"EVALUATING {split_name.upper()} SET")
    print(f"{'=' * 80}")
    
    all_preds = {label: [] for label in label_names}
    all_probs = {label: [] for label in label_names}
    all_labels = {label: [] for label in label_names}
    
    print(f"\nRunning inference on {len(dataset)} samples...")
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            features = sample[0]
            labels = sample[1]
            
            # Prepare input batch
            x_batch = {}
            for k, v in features.items():
                if isinstance(v, torch.Tensor):
                    x_batch[k] = v.unsqueeze(0).to(device)
                else:
                    x_batch[k] = torch.tensor([[v]], dtype=torch.float32).to(device)
            
            # Forward pass
            try:
                outputs = model.net_(x_batch, None)
                
                # Collect predictions
                for label_name in label_names:
                    if label_name in outputs:
                        prob = torch.sigmoid(outputs[label_name]).cpu().item()
                        pred = 1 if prob > 0.5 else 0
                        
                        all_preds[label_name].append(pred)
                        all_probs[label_name].append(prob)
                        all_labels[label_name].append(labels[label_name])
            except Exception as e:
                print(f"  Error on sample {i}: {e}")
                continue
            
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(dataset)} samples...")
    
    print(f"✓ Inference complete")
    
    # Calculate metrics
    print(f"\nCalculating metrics...")
    
    metrics = {
        'metadata': {
            'split': split_name,
            'num_samples': len(dataset),
            'num_classes': len(label_names),
            'class_names': label_names,
            'evaluation_time': datetime.now().isoformat()
        },
        'overall': {},
        'per_class': {}
    }
    
    # Per-class metrics
    for label_name in label_names:
        y_true = np.array(all_labels[label_name])
        y_pred = np.array(all_preds[label_name])
        y_prob = np.array(all_probs[label_name])
        
        if len(y_true) == 0 or y_true.sum() == 0:
            continue
        
        # Basic metrics
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        # AUC scores
        try:
            auc_roc = roc_auc_score(y_true, y_prob)
        except:
            auc_roc = 0.0
        
        try:
            auc_pr = average_precision_score(y_true, y_prob)
        except:
            auc_pr = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        metrics['per_class'][label_name] = {
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'support': int(y_true.sum()),
            'total_samples': int(len(y_true)),
            'auc_roc': float(auc_roc),
            'auc_pr': float(auc_pr),
            'confusion_matrix': cm.tolist()
        }
    
    # Overall metrics
    all_y_true = []
    all_y_pred = []
    for label_name in label_names:
        if label_name in metrics['per_class']:
            all_y_true.extend(all_labels[label_name])
            all_y_pred.extend(all_preds[label_name])
    
    if len(all_y_true) > 0:
        metrics['overall']['accuracy'] = float(accuracy_score(all_y_true, all_y_pred))
        metrics['overall']['balanced_accuracy'] = float(balanced_accuracy_score(all_y_true, all_y_pred))
        
        # Macro averages
        precisions = [m['precision'] for m in metrics['per_class'].values()]
        recalls = [m['recall'] for m in metrics['per_class'].values()]
        f1s = [m['f1_score'] for m in metrics['per_class'].values()]
        
        metrics['overall']['macro_precision'] = float(np.mean(precisions))
        metrics['overall']['macro_recall'] = float(np.mean(recalls))
        metrics['overall']['macro_f1'] = float(np.mean(f1s))
    
    print(f"✓ Metrics calculated")
    
    # Print summary
    print(f"\n{'-' * 80}")
    print(f"RESULTS SUMMARY - {split_name.upper()}")
    print(f"{'-' * 80}")
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {metrics['overall'].get('accuracy', 0):.4f}")
    print(f"  Balanced Accuracy: {metrics['overall'].get('balanced_accuracy', 0):.4f}")
    print(f"  Macro F1: {metrics['overall'].get('macro_f1', 0):.4f}")
    
    print(f"\nPer-Class Performance:")
    print(f"{'Class':<10} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AUC-ROC':<10} {'Support':<10}")
    print(f"{'-' * 80}")
    for label_name, class_metrics in metrics['per_class'].items():
        print(f"{label_name:<10} "
              f"{class_metrics['accuracy']:<8.4f} "
              f"{class_metrics['precision']:<8.4f} "
              f"{class_metrics['recall']:<8.4f} "
              f"{class_metrics['f1_score']:<8.4f} "
              f"{class_metrics['auc_roc']:<10.4f} "
              f"{class_metrics['support']:<10}")
    
    # Save metrics
    output_file = f"{split_name}_metrics.json"
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Metrics saved to: {output_file}")
    
    return metrics

# Evaluate all splits
all_results = {}
for split_name in ['train', 'validation', 'test']:
    all_results[split_name] = evaluate_split(model, datasets[split_name], split_name)

print(f"\n{'=' * 80}")
print("EVALUATION COMPLETE")
print(f"{'=' * 80}")
print(f"\nAll metrics saved:")
print(f"  - train_metrics.json")
print(f"  - validation_metrics.json")
print(f"  - test_metrics.json")

