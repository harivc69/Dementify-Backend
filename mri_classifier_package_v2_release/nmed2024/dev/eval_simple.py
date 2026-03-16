#!/usr/bin/env python3
"""
Simple evaluation script using ADRDModel's predict_proba method
"""

import sys
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset_csv import CSVDataset
from adrd.model import ADRDModel

def evaluate_split(model, dataset, split_name):
    """Evaluate model on a dataset split"""
    print(f"\n{'=' * 80}")
    print(f"EVALUATING {split_name.upper()} SET ({len(dataset)} samples)")
    print(f"{'=' * 80}")
    
    # Get predictions
    print(f"\nGetting predictions...")
    logits, probas = model.predict_proba(dataset.features, _batch_size=32)
    print(f"✓ Got predictions")
    
    # Extract labels
    label_names = list(dataset.tgt_modalities.keys())
    
    # Calculate metrics per class
    metrics = {'per_class': {}, 'overall': {}}
    
    for label_name in label_names:
        y_true = np.array([sample[label_name] for sample in dataset.labels])
        y_prob = np.array([prob[label_name] for prob in probas])
        y_pred = (y_prob > 0.5).astype(int)
        
        if y_true.sum() == 0:
            continue
        
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        
        try:
            auc_roc = roc_auc_score(y_true, y_prob)
        except:
            auc_roc = 0.0
        
        try:
            auc_pr = average_precision_score(y_true, y_prob)
        except:
            auc_pr = 0.0
        
        metrics['per_class'][label_name] = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'support': int(y_true.sum()),
            'auc_roc': float(auc_roc),
            'auc_pr': float(auc_pr)
        }
    
    # Overall metrics
    all_y_true = []
    all_y_pred = []
    for label_name in label_names:
        if label_name in metrics['per_class']:
            y_true = np.array([sample[label_name] for sample in dataset.labels])
            y_prob = np.array([prob[label_name] for prob in probas])
            y_pred = (y_prob > 0.5).astype(int)
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
    
    metrics['overall']['accuracy'] = float(accuracy_score(all_y_true, all_y_pred))
    
    precisions = [m['precision'] for m in metrics['per_class'].values()]
    recalls = [m['recall'] for m in metrics['per_class'].values()]
    f1s = [m['f1_score'] for m in metrics['per_class'].values()]
    
    metrics['overall']['macro_precision'] = float(np.mean(precisions))
    metrics['overall']['macro_recall'] = float(np.mean(recalls))
    metrics['overall']['macro_f1'] = float(np.mean(f1s))
    
    # Print summary
    print(f"\nOverall: Acc={metrics['overall']['accuracy']:.4f}, Macro-F1={metrics['overall']['macro_f1']:.4f}")
    print(f"\nPer-Class:")
    print(f"{'Class':<10} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AUC':<8} {'Supp':<8}")
    print("-" * 70)
    for label, m in metrics['per_class'].items():
        print(f"{label:<10} {m['accuracy']:.4f}   {m['precision']:.4f}   {m['recall']:.4f}   {m['f1_score']:.4f}   {m['auc_roc']:.4f}   {m['support']}")
    
    # Save
    with open(f"{split_name}_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Saved to {split_name}_metrics.json")
    
    return metrics

print("=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

# Load datasets first
datasets = {}
for split, path, mode in [
    ('train', 'data/processed/train.csv', 0),
    ('validation', 'data/processed/val.csv', 1),
    ('test', 'data/processed/test.csv', 2)
]:
    print(f"\nLoading {split} dataset...")
    datasets[split] = CSVDataset(
        dat_file=path,
        cnf_file='data/toml_files/custom_nacc_config_no_img.toml',
        mode=mode,
        img_mode=-1,
        arch='NonImg',
        transforms=None,
        stripped=None
    )
    print(f"✓ {split}: {len(datasets[split])} samples")

# Load model
print("\nLoading model...")
import torch
checkpoint = torch.load('checkpoints_overnight/final_model_epoch_150.pt', map_location='cpu')
config = checkpoint['config']

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
    device='cuda:0' if torch.cuda.is_available() else 'cpu',
    cuda_devices=[0] if torch.cuda.is_available() else [],
    img_net='NonImg',
    verbose=0
)

# Initialize and load weights
model._init_net()
model.net_.load_state_dict(checkpoint['model_state_dict'])
model.net_.eval()
print("✓ Model loaded and ready")

# Evaluate
results = {}
for split in ['train', 'validation', 'test']:
    results[split] = evaluate_split(model, datasets[split], split)

print(f"\n{'=' * 80}")
print("EVALUATION COMPLETE")
print(f"{'=' * 80}")

