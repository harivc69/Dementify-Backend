#!/usr/bin/env python3
"""
MRI Explainability CLI

Generate GradCAM and SHAP explanations for MRI classification models.

Usage:
    # GradCAM explanation
    python explain_mri.py \
        --checkpoint checkpoints/stage1_balanced_lightmri/model_best.pt \
        --input preprocessed_mri/NACC_S_001341_20170307.npy \
        --method gradcam \
        --output_dir explanations/subject_001341/

    # SHAP explanation
    python explain_mri.py \
        --checkpoint checkpoints/stage1_balanced_lightmri/model_best.pt \
        --input preprocessed_mri/NACC_S_001341_20170307.npy \
        --method shap \
        --output_dir explanations/subject_001341/

    # Both methods
    python explain_mri.py \
        --checkpoint checkpoints/stage1_balanced_lightmri/model_best.pt \
        --input preprocessed_mri/NACC_S_001341_20170307.npy \
        --method both \
        --output_dir explanations/subject_001341/
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from datetime import datetime

# Add paths for imports (relative to this file's location)
_EXPLAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_MRI_TRAINING_DIR = os.path.dirname(_EXPLAIN_DIR)
_NMED_DEV = os.path.join(_MRI_TRAINING_DIR, 'nmed2024', 'dev')
_NMED_ROOT = os.path.join(_MRI_TRAINING_DIR, 'nmed2024')
if _NMED_DEV not in sys.path:
    sys.path.insert(0, _NMED_DEV)
if _NMED_ROOT not in sys.path:
    sys.path.insert(0, _NMED_ROOT)

from .validation import ExplainabilityValidator, run_all_sanity_checks
from .nifti_mapper import NIfTIMapper
from .reverse_transform import ReverseTransformPipeline
from .gradcam_mri import GradCAMMRI, compute_gradcam_for_sample
from .shap_mri import SHAPExplainerMRI, compute_shap_for_sample
from .visualization import ExplainabilityVisualizer, create_comparison_figure


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate MRI explainability visualizations'
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--input', type=str, required=True,
        help='Path to preprocessed MRI file (.npy)'
    )
    parser.add_argument(
        '--method', type=str, default='gradcam',
        choices=['gradcam', 'shap', 'both'],
        help='Explanation method to use'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--target_class', type=int, default=None,
        help='Target class/embedding index to explain (default: auto-select argmax)'
    )
    parser.add_argument(
        '--target_layer', type=str, default='bn4',
        help='Target layer for GradCAM (default: bn4)'
    )
    parser.add_argument(
        '--supervoxel_size', type=int, default=8,
        help='Supervoxel size for SHAP (default: 8)'
    )
    parser.add_argument(
        '--n_shap_samples', type=int, default=100,
        help='Number of samples for SHAP estimation (default: 100)'
    )
    parser.add_argument(
        '--n_slices', type=int, default=5,
        help='Number of PNG slices per view (default: 5)'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.2,
        help='Attribution threshold for overlay (default: 0.2)'
    )
    parser.add_argument(
        '--strict', action='store_true',
        help='Enable strict validation (fail on any validation error)'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to use (cuda or cpu)'
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str):
    """Load trained LightMRI3D model from ADRDModel checkpoint.

    The ADRDModel saves an OrderedDict where the MRI encoder weights live
    under the prefix ``modules_emb_src.img_MRI_T1_1.img_model.*``.
    We extract those weights and load them into a fresh LightMRI3D.
    """
    from adrd.nn.img_model_wrapper import LightMRI3D

    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = LightMRI3D(in_channels=1, out_dim=128, dropout=0.3)

    # ADRDModel format: OrderedDict with full key paths
    prefix = 'modules_emb_src.img_MRI_T1_1.img_model.'
    if isinstance(checkpoint, dict) and any(k.startswith(prefix) for k in checkpoint):
        img_state = {
            k[len(prefix):]: v
            for k, v in checkpoint.items()
            if k.startswith(prefix)
        }
        model.load_state_dict(img_state)
        print(f"Loaded LightMRI3D weights ({len(img_state)} params) from ADRDModel checkpoint")
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded LightMRI3D from state_dict key")
    else:
        try:
            model.load_state_dict(checkpoint)
            print("Loaded LightMRI3D from direct state dict")
        except Exception:
            print("WARNING: Could not load weights, using random initialization")

    model.to(device)
    model.eval()
    return model


def run_gradcam(
    model: torch.nn.Module,
    input_path: str,
    nifti_mapper: NIfTIMapper,
    output_dir: str,
    args
) -> dict:
    """Run GradCAM explanation pipeline."""
    print("\n" + "=" * 60)
    print("Running GradCAM")
    print("=" * 60)

    # Load input
    input_data = np.load(input_path)
    input_tensor = torch.from_numpy(input_data).float()
    if input_tensor.dim() == 4:
        input_tensor = input_tensor.unsqueeze(0)

    # Initialize GradCAM
    gradcam = GradCAMMRI(
        model,
        target_layer=args.target_layer,
        device=args.device
    )

    # Compute GradCAM
    print(f"Computing GradCAM on layer: {args.target_layer}")
    result = gradcam.compute(input_tensor, target_class=args.target_class)
    cam_tensor = result.cam

    print(f"Raw CAM shape: {cam_tensor.shape}")
    print(f"CAM range: [{cam_tensor.min():.4f}, {cam_tensor.max():.4f}]")

    # Get original NIfTI info
    try:
        original_nifti, affine, original_shape = nifti_mapper.get_original(input_path)
        original_data = original_nifti.get_fdata()
        if original_data.ndim == 4:
            original_data = original_data[..., 0]
        print(f"Original shape: {original_shape}")
    except FileNotFoundError as e:
        print(f"Warning: Could not find original NIfTI: {e}")
        print("Using preprocessed space for visualization")
        original_nifti = None
        affine = np.eye(4)
        original_shape = (128, 128, 128)
        original_data = input_data[0] if input_data.ndim == 4 else input_data

    # Reverse transform
    print("Applying reverse transform...")
    pipeline = ReverseTransformPipeline(
        strict_validation=args.strict,
        interpolation_order=1
    )

    attribution, validation_report = pipeline.transform(
        cam_tensor,
        original_shape,
        affine,
        original_data
    )

    print(f"Transformed attribution shape: {attribution.shape}")

    # Additional sanity checks
    sanity_report = run_all_sanity_checks(
        attribution, original_data, original_shape, affine, strict=False
    )
    validation_report['sanity_checks'] = sanity_report

    # Save visualizations
    print("\nGenerating visualizations...")
    gradcam_dir = os.path.join(output_dir, 'gradcam')
    visualizer = ExplainabilityVisualizer(
        threshold=args.threshold,
        overlay_alpha=0.6
    )

    vis_result = visualizer.save_all(
        attribution,
        affine,
        gradcam_dir,
        original_nifti=original_nifti,
        validation_report=validation_report,
        n_slices=args.n_slices
    )

    return {
        'method': 'gradcam',
        'attribution': attribution,
        'affine': affine,
        'original_shape': original_shape,
        'validation_report': validation_report,
        'output_dir': gradcam_dir,
        'vis_result': vis_result,
    }


def run_shap(
    model: torch.nn.Module,
    input_path: str,
    nifti_mapper: NIfTIMapper,
    output_dir: str,
    args
) -> dict:
    """Run SHAP explanation pipeline."""
    print("\n" + "=" * 60)
    print("Running SHAP")
    print("=" * 60)

    # Load input
    input_data = np.load(input_path)
    input_tensor = torch.from_numpy(input_data).float()
    if input_tensor.dim() == 4:
        input_tensor = input_tensor.unsqueeze(0)

    # Initialize SHAP explainer
    print(f"Initializing SHAP with supervoxel size: {args.supervoxel_size}")
    explainer = SHAPExplainerMRI(
        model,
        supervoxel_size=args.supervoxel_size,
        device=args.device
    )

    # Compute SHAP values
    print(f"Computing SHAP values ({args.n_shap_samples} samples)...")
    result = explainer.compute(input_tensor, n_samples=args.n_shap_samples)
    shap_values = result.shap_values

    print(f"SHAP values shape: {shap_values.shape}")
    print(f"SHAP range: [{shap_values.min():.4f}, {shap_values.max():.4f}]")
    print(f"Base value: {result.base_value:.4f}")
    print(f"Prediction: {result.prediction:.4f}")

    # Get original NIfTI info
    try:
        original_nifti, affine, original_shape = nifti_mapper.get_original(input_path)
        original_data = original_nifti.get_fdata()
        if original_data.ndim == 4:
            original_data = original_data[..., 0]
        print(f"Original shape: {original_shape}")
    except FileNotFoundError as e:
        print(f"Warning: Could not find original NIfTI: {e}")
        original_nifti = None
        affine = np.eye(4)
        original_shape = (128, 128, 128)
        original_data = input_data[0] if input_data.ndim == 4 else input_data

    # Reverse transform
    print("Applying reverse transform...")
    shap_tensor = torch.from_numpy(shap_values).float()

    pipeline = ReverseTransformPipeline(
        strict_validation=args.strict,
        interpolation_order=1
    )

    attribution, validation_report = pipeline.transform(
        shap_tensor,
        original_shape,
        affine,
        original_data
    )

    print(f"Transformed attribution shape: {attribution.shape}")

    # Save visualizations
    print("\nGenerating visualizations...")
    shap_dir = os.path.join(output_dir, 'shap')
    visualizer = ExplainabilityVisualizer(
        threshold=args.threshold,
        overlay_alpha=0.6
    )

    vis_result = visualizer.save_all(
        attribution,
        affine,
        shap_dir,
        original_nifti=original_nifti,
        validation_report=validation_report,
        n_slices=args.n_slices
    )

    return {
        'method': 'shap',
        'attribution': attribution,
        'affine': affine,
        'original_shape': original_shape,
        'validation_report': validation_report,
        'output_dir': shap_dir,
        'vis_result': vis_result,
        'base_value': result.base_value,
        'prediction': result.prediction,
    }


def main():
    args = parse_args()

    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Setup output directory
    if args.output_dir is None:
        basename = os.path.basename(args.input).replace('.npy', '')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'explanations/{basename}_{timestamp}'

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Load model
    model = load_model(args.checkpoint, args.device)

    # Initialize NIfTI mapper
    nifti_mapper = NIfTIMapper()

    # Run explanations
    results = {}

    if args.method in ['gradcam', 'both']:
        results['gradcam'] = run_gradcam(
            model, args.input, nifti_mapper, args.output_dir, args
        )

    if args.method in ['shap', 'both']:
        results['shap'] = run_shap(
            model, args.input, nifti_mapper, args.output_dir, args
        )

    # Create comparison if both methods used
    if args.method == 'both' and 'gradcam' in results and 'shap' in results:
        print("\n" + "=" * 60)
        print("Creating comparison visualization")
        print("=" * 60)

        try:
            original_nifti, _, original_shape = nifti_mapper.get_original(args.input)
            original_data = original_nifti.get_fdata()
            if original_data.ndim == 4:
                original_data = original_data[..., 0]

            # Mid-slice comparison
            mid_idx = original_shape[2] // 2

            comparison_path = os.path.join(args.output_dir, 'comparison.png')
            create_comparison_figure(
                original_data,
                results['gradcam']['attribution'],
                results['shap']['attribution'],
                slice_idx=mid_idx,
                axis=2,
                output_path=comparison_path
            )
        except Exception as e:
            print(f"Could not create comparison: {e}")

    # Save summary
    summary = {
        'input': args.input,
        'checkpoint': args.checkpoint,
        'method': args.method,
        'target_class': args.target_class,
        'timestamp': datetime.now().isoformat(),
        'device': args.device,
        'methods_run': list(results.keys()),
    }

    for method, result in results.items():
        summary[f'{method}_validation_passed'] = \
            result['validation_report'].get('all_passed', False)

    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Output saved to: {args.output_dir}")
    print(f"Summary: {summary_path}")

    for method in results:
        print(f"\n{method.upper()}:")
        print(f"  Output: {results[method]['output_dir']}")
        print(f"  Validation passed: {results[method]['validation_report'].get('all_passed', False)}")


if __name__ == '__main__':
    main()
