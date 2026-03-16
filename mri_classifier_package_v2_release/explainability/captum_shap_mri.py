#!/usr/bin/env python3
"""
GPU-Accelerated SHAP for MRI using Captum.

Uses GradientShap (expected gradients) for ~400x speedup over permutation SHAP.
Produces voxel-level (128³) attribution maps in 5-15 seconds per patient.

Methods available:
- GradientShap: SHAP-grounded, fast, good quality
- DeepLiftShap: Exact SHAP via DeepLIFT rules, very fast
- IntegratedGradients: Axiomatic, best for neuroanatomic localization
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

# Add paths for imports (relative to this file's location)
_EXPLAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_MRI_TRAINING_DIR = os.path.dirname(_EXPLAIN_DIR)
_NMED_DEV = os.path.join(_MRI_TRAINING_DIR, 'nmed2024', 'dev')
_NMED_ROOT = os.path.join(_MRI_TRAINING_DIR, 'nmed2024')
if _MRI_TRAINING_DIR not in sys.path:
    sys.path.insert(0, _MRI_TRAINING_DIR)
if _NMED_DEV not in sys.path:
    sys.path.insert(0, _NMED_DEV)
if _NMED_ROOT not in sys.path:
    sys.path.insert(0, _NMED_ROOT)

from captum.attr import GradientShap, DeepLiftShap, IntegratedGradients, NoiseTunnel
from mri_training.explainability.explain_mri import load_model


class MRIModelWrapper(nn.Module):
    """
    Wrapper around LightMRI3D that returns a single scalar output
    suitable for Captum attribution methods.

    LightMRI3D outputs (batch, out_dim=128) embedding.
    We reduce to a single scalar via mean projection, which represents
    the overall "activation" of the MRI encoder for the target class.

    For class-specific attribution, we can use different projection heads.
    """
    def __init__(self, mri_model, projection='norm'):
        super().__init__()
        self.model = mri_model
        self.projection = projection

    def forward(self, x):
        """
        Args:
            x: (B, 1, 128, 128, 128) MRI volume

        Returns:
            (B, 1) scalar output
        """
        emb = self.model(x)  # (B, 128)

        if self.projection == 'norm':
            # L2 norm of embedding — measures overall MRI signal strength
            return emb.norm(dim=1, keepdim=True)
        elif self.projection == 'mean':
            return emb.mean(dim=1, keepdim=True)
        elif self.projection == 'first':
            # First dimension (arbitrary but deterministic)
            return emb[:, 0:1]
        else:
            return emb.norm(dim=1, keepdim=True)


def compute_gradient_shap(
    model,
    input_tensor,
    n_samples=50,
    n_baselines=10,
    stdevs=0.09,
    device='cuda'
):
    """
    Compute GradientShap attribution for MRI volume.

    GradientShap approximates SHAP values using expected gradients:
    E[∂f/∂x * (x - x')] averaged over random baselines x' and noise.

    Args:
        model: LightMRI3D model
        input_tensor: (1, 1, 128, 128, 128)
        n_samples: Number of interpolation samples (more = smoother)
        n_baselines: Number of random baselines
        stdevs: Gaussian noise std for smoothing
        device: cuda or cpu

    Returns:
        attribution: (128, 128, 128) numpy array, normalized to [0, 1]
        elapsed: time in seconds
    """
    wrapped = MRIModelWrapper(model, projection='norm').to(device).eval()
    gs = GradientShap(wrapped)

    # Baselines: zeros (represents "no MRI signal")
    baselines = torch.zeros(n_baselines, 1, 128, 128, 128, device=device)

    input_tensor = input_tensor.to(device)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    elif input_tensor.dim() == 4:
        input_tensor = input_tensor.unsqueeze(0)

    input_tensor.requires_grad_(True)

    start = time.time()
    attr = gs.attribute(
        input_tensor,
        baselines=baselines,
        n_samples=n_samples,
        stdevs=stdevs,
    )
    elapsed = time.time() - start

    # Convert to numpy and squeeze
    attr_np = attr.detach().cpu().numpy()
    while attr_np.ndim > 3:
        attr_np = attr_np[0]

    # Take absolute value (both positive and negative contributions matter for localization)
    attr_np = np.abs(attr_np)

    # Normalize to [0, 1]
    a_min, a_max = attr_np.min(), attr_np.max()
    if a_max - a_min > 1e-8:
        attr_np = (attr_np - a_min) / (a_max - a_min)

    return attr_np, elapsed


def compute_deep_lift_shap(
    model,
    input_tensor,
    n_baselines=10,
    device='cuda'
):
    """
    Compute DeepLiftShap attribution.

    Uses DeepLIFT propagation rules with multiple baselines to approximate SHAP.
    Fastest method (~2-5 seconds).
    """
    wrapped = MRIModelWrapper(model, projection='norm').to(device).eval()
    dls = DeepLiftShap(wrapped)

    baselines = torch.zeros(n_baselines, 1, 128, 128, 128, device=device)

    input_tensor = input_tensor.to(device)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    elif input_tensor.dim() == 4:
        input_tensor = input_tensor.unsqueeze(0)

    start = time.time()
    attr = dls.attribute(input_tensor, baselines=baselines)
    elapsed = time.time() - start

    attr_np = attr.detach().cpu().numpy()
    while attr_np.ndim > 3:
        attr_np = attr_np[0]

    attr_np = np.abs(attr_np)
    a_min, a_max = attr_np.min(), attr_np.max()
    if a_max - a_min > 1e-8:
        attr_np = (attr_np - a_min) / (a_max - a_min)

    return attr_np, elapsed


def compute_integrated_gradients(
    model,
    input_tensor,
    n_steps=50,
    device='cuda'
):
    """
    Compute Integrated Gradients attribution.

    Axiomatic method: integrates gradients along path from baseline to input.
    Best for neuroanatomic localization per BrainAge literature.
    """
    wrapped = MRIModelWrapper(model, projection='norm').to(device).eval()
    ig = IntegratedGradients(wrapped)

    baseline = torch.zeros(1, 1, 128, 128, 128, device=device)

    input_tensor = input_tensor.to(device)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    elif input_tensor.dim() == 4:
        input_tensor = input_tensor.unsqueeze(0)

    start = time.time()
    attr = ig.attribute(
        input_tensor,
        baselines=baseline,
        n_steps=n_steps,
        internal_batch_size=10,  # Process 10 interpolation steps at a time
    )
    elapsed = time.time() - start

    attr_np = attr.detach().cpu().numpy()
    while attr_np.ndim > 3:
        attr_np = attr_np[0]

    attr_np = np.abs(attr_np)
    a_min, a_max = attr_np.min(), attr_np.max()
    if a_max - a_min > 1e-8:
        attr_np = (attr_np - a_min) / (a_max - a_min)

    return attr_np, elapsed


def compute_smooth_integrated_gradients(
    model,
    input_tensor,
    n_steps=50,
    nt_samples=5,
    stdevs=0.05,
    device='cuda'
):
    """
    Compute SmoothGrad + Integrated Gradients.

    Adds Gaussian noise averaging on top of IG for smoother attributions.
    Best quality but slowest of the Captum methods (~30-60 seconds).
    """
    wrapped = MRIModelWrapper(model, projection='norm').to(device).eval()
    ig = IntegratedGradients(wrapped)
    nt = NoiseTunnel(ig)

    baseline = torch.zeros(1, 1, 128, 128, 128, device=device)

    input_tensor = input_tensor.to(device)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    elif input_tensor.dim() == 4:
        input_tensor = input_tensor.unsqueeze(0)

    start = time.time()
    attr = nt.attribute(
        input_tensor,
        baselines=baseline,
        nt_type='smoothgrad',
        nt_samples=nt_samples,
        stdevs=stdevs,
        n_steps=n_steps,
        internal_batch_size=10,
    )
    elapsed = time.time() - start

    attr_np = attr.detach().cpu().numpy()
    while attr_np.ndim > 3:
        attr_np = attr_np[0]

    attr_np = np.abs(attr_np)
    a_min, a_max = attr_np.min(), attr_np.max()
    if a_max - a_min > 1e-8:
        attr_np = (attr_np - a_min) / (a_max - a_min)

    return attr_np, elapsed


# Convenience mapping
METHODS = {
    'gradient_shap': compute_gradient_shap,
    'deep_lift_shap': compute_deep_lift_shap,
    'integrated_gradients': compute_integrated_gradients,
    'smooth_ig': compute_smooth_integrated_gradients,
}


if __name__ == '__main__':
    """Quick test: run all methods on a sample scan."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--method', type=str, default='all',
                        choices=['gradient_shap', 'deep_lift_shap',
                                 'integrated_gradients', 'smooth_ig', 'all'])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device)
    input_data = np.load(args.input)
    input_tensor = torch.from_numpy(input_data).float()

    methods_to_run = METHODS.keys() if args.method == 'all' else [args.method]

    for method_name in methods_to_run:
        print(f"\n{'='*50}")
        print(f"Method: {method_name}")
        print(f"{'='*50}")

        method_fn = METHODS[method_name]
        attr, elapsed = method_fn(model, input_tensor, device=args.device)

        print(f"  Time: {elapsed:.1f}s")
        print(f"  Shape: {attr.shape}")
        print(f"  Range: [{attr.min():.4f}, {attr.max():.4f}]")
        print(f"  Nonzero (>0.1): {(attr > 0.1).sum()} / {attr.size} ({100*(attr>0.1).sum()/attr.size:.1f}%)")
