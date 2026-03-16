#!/usr/bin/env python3
"""
Improved MRI Heatmap Overlay Generation.

Iteratively refined visualization with:
1. Higher resolution target layer (bn3: 16x16x16 vs bn4: 8x8x8)
2. Cubic spline upsampling for smoother gradients
3. Gaussian smoothing to reduce blockiness
4. Brain masking to restrict heatmap to brain tissue
5. Percentile-based thresholding for better contrast
6. GradCAM++ for multi-region attribution
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from scipy.ndimage import zoom, gaussian_filter, binary_fill_holes, binary_dilation, binary_erosion
from scipy.ndimage import label as ndimage_label, generate_binary_structure


def make_heatmap_cmap():
    """
    Create a clinical neuroimaging colormap: transparent → red → yellow → white.

    This provides better contrast against the gray MRI background than inferno,
    and matches the red-yellow convention used in SPM/FSL neuroimaging tools.
    """
    from matplotlib.colors import LinearSegmentedColormap
    colors = [
        (0.0, 0.0, 0.0, 0.0),   # fully transparent (below threshold)
        (0.6, 0.0, 0.0, 0.85),   # dark red
        (1.0, 0.0, 0.0, 0.90),   # red
        (1.0, 0.5, 0.0, 0.92),   # orange
        (1.0, 1.0, 0.0, 0.95),   # yellow
        (1.0, 1.0, 1.0, 1.0),    # white (peak)
    ]
    return LinearSegmentedColormap.from_list('hot_clinical', colors, N=256)


HEATMAP_CMAP = make_heatmap_cmap()

# Add paths for imports (relative to this file's location)
_EXPLAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_MRI_TRAINING_DIR = os.path.dirname(_EXPLAIN_DIR)
_NMED_DEV = os.path.join(_MRI_TRAINING_DIR, 'nmed2024', 'dev')
_NMED_ROOT = os.path.join(_MRI_TRAINING_DIR, 'nmed2024')
if _NMED_DEV not in sys.path:
    sys.path.insert(0, _NMED_DEV)
if _NMED_ROOT not in sys.path:
    sys.path.insert(0, _NMED_ROOT)

from mri_training.explainability.gradcam_mri import GradCAMMRI
from mri_training.explainability.explain_mri import load_model


def create_brain_mask(mri_data: np.ndarray, mask_path: str = None) -> np.ndarray:
    """
    Create a brain mask, preferring pre-computed deepbet mask.

    Priority:
    1. Load pre-computed *_mask.npy from skull stripping (most accurate)
    2. Intensity threshold fallback (for skull-stripped data, zeros = non-brain)

    Args:
        mri_data: 3D MRI volume (128x128x128)
        mask_path: Path to pre-computed mask .npy file

    Returns:
        Boolean 3D brain mask
    """
    # Try loading pre-computed mask from skull stripping
    if mask_path is not None and os.path.exists(mask_path):
        mask = np.load(mask_path)
        if mask.shape == mri_data.shape:
            return mask.astype(bool)

    # Fallback: intensity threshold
    # For skull-stripped data, non-brain voxels are ~0 so this works well
    d_min, d_max = mri_data.min(), mri_data.max()
    if d_max - d_min < 1e-8:
        return np.ones_like(mri_data, dtype=bool)
    mri_norm = (mri_data - d_min) / (d_max - d_min)

    # Threshold + light morphological cleanup
    mask = mri_norm > 0.02
    struct = generate_binary_structure(3, 1)
    mask = binary_fill_holes(mask)
    mask = binary_erosion(mask, structure=struct, iterations=1)
    mask = binary_dilation(mask, structure=struct, iterations=1)

    return mask


def compute_improved_gradcam(
    model,
    input_tensor: torch.Tensor,
    target_layer: str = 'bn3',
    target_class: int = 0,
    device: str = 'cuda',
    use_gradcam_pp: bool = True
) -> np.ndarray:
    """Compute GradCAM with improved settings."""
    gradcam = GradCAMMRI(
        model,
        target_layer=target_layer,
        device=device,
        use_gradcam_pp=use_gradcam_pp
    )
    result = gradcam.compute(input_tensor, target_class=target_class)
    cam = result.cam.cpu().numpy()

    # Squeeze to 3D
    while cam.ndim > 3:
        cam = cam[0]

    print(f"  Raw CAM shape: {cam.shape}")
    print(f"  Raw CAM range: [{cam.min():.4f}, {cam.max():.4f}]")
    print(f"  Raw CAM nonzero: {(cam > 0.01).sum()} / {cam.size}")

    return cam


def upsample_and_smooth(
    cam: np.ndarray,
    target_shape: tuple = (128, 128, 128),
    interpolation_order: int = 3,
    sigma: float = 2.0
) -> np.ndarray:
    """Upsample CAM to target shape with cubic interpolation + Gaussian smoothing."""
    # Cubic spline upsampling
    zoom_factors = tuple(t / c for t, c in zip(target_shape, cam.shape))
    cam_upsampled = zoom(cam, zoom_factors, order=interpolation_order)

    # Gaussian smoothing to reduce blockiness
    if sigma > 0:
        cam_smooth = gaussian_filter(cam_upsampled, sigma=sigma)
    else:
        cam_smooth = cam_upsampled

    # Normalize to [0, 1]
    c_min, c_max = cam_smooth.min(), cam_smooth.max()
    if c_max - c_min > 1e-8:
        cam_smooth = (cam_smooth - c_min) / (c_max - c_min)

    return cam_smooth


def apply_brain_mask(attribution: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    """Zero out attribution outside the brain."""
    masked = attribution.copy()
    masked[~brain_mask] = 0
    return masked


def save_improved_slices(
    mri_data: np.ndarray,
    attribution: np.ndarray,
    output_dir: str,
    threshold_percentile: float = 85,
    alpha: float = 0.7,
    colormap: str = 'hot_clinical',
    n_slices: int = 5
):
    """
    Save improved overlay slices.

    Uses percentile-based thresholding and better visualization parameters.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Normalize MRI for display
    d_min, d_max = mri_data.min(), mri_data.max()
    if d_max > d_min:
        mri_display = (mri_data - d_min) / (d_max - d_min)
    else:
        mri_display = np.zeros_like(mri_data)

    # Compute threshold from percentile of nonzero values
    nonzero_attr = attribution[attribution > 0.01]
    if len(nonzero_attr) > 0:
        threshold = np.percentile(nonzero_attr, threshold_percentile)
    else:
        threshold = 0.2
    print(f"  Using threshold: {threshold:.4f} (percentile {threshold_percentile})")

    views = {
        'axial': (2, 'z'),
        'coronal': (1, 'y'),
        'sagittal': (0, 'x'),
    }

    png_paths = []
    for view_name, (axis, label) in views.items():
        view_dir = os.path.join(output_dir, view_name)
        os.makedirs(view_dir, exist_ok=True)

        n_total = mri_data.shape[axis]
        start = n_total // 5
        end = n_total * 4 // 5
        indices = np.linspace(start, end, n_slices, dtype=int)

        for i, idx in enumerate(indices):
            orig_slice = np.rot90(np.take(mri_display, idx, axis=axis))
            attr_slice = np.rot90(np.take(attribution, idx, axis=axis))

            fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='black')

            # Panel 1: Original MRI
            axes[0].imshow(orig_slice, cmap='gray', vmin=0, vmax=1)
            axes[0].set_title(f'MRI ({label}={idx})', fontsize=14, color='white')
            axes[0].axis('off')

            # Panel 2: Attribution heatmap
            cmap_obj = HEATMAP_CMAP if colormap == 'hot_clinical' else colormap
            im = axes[1].imshow(attr_slice, cmap=cmap_obj, vmin=0, vmax=1)
            axes[1].set_title('Attribution', fontsize=14, color='white')
            axes[1].axis('off')
            cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

            # Panel 3: Overlay with percentile threshold
            axes[2].imshow(orig_slice, cmap='gray', vmin=0, vmax=1)
            masked_attr = np.ma.masked_where(attr_slice < threshold, attr_slice)
            axes[2].imshow(
                masked_attr, cmap=cmap_obj,
                alpha=alpha, vmin=threshold, vmax=1.0
            )
            axes[2].set_title(f'Overlay (top {100-threshold_percentile:.0f}%)', fontsize=14, color='white')
            axes[2].axis('off')

            plt.tight_layout()
            png_path = os.path.join(view_dir, f'slice_{i:02d}_{label}{idx}.png')
            plt.savefig(png_path, dpi=150, bbox_inches='tight',
                       facecolor='black', edgecolor='none')
            plt.close()
            png_paths.append(png_path)

    print(f"  Saved {len(png_paths)} PNG slices to {output_dir}")
    return png_paths


def run_improved_heatmap(
    checkpoint_path: str,
    input_npy_path: str,
    output_dir: str,
    target_layer: str = 'bn3',
    target_class: int = 0,
    use_gradcam_pp: bool = True,
    interpolation_order: int = 3,
    sigma: float = 2.0,
    threshold_percentile: float = 85,
    alpha: float = 0.7,
    colormap: str = 'hot_clinical',
    device: str = 'cuda',
    n_slices: int = 5,
    mask_path: str = None
):
    """Run the full improved heatmap pipeline."""
    print("=" * 60)
    print("Improved GradCAM Heatmap Generation")
    print("=" * 60)

    # Load model
    print(f"\n1. Loading model from {checkpoint_path}")
    model = load_model(checkpoint_path, device)

    # Load input
    print(f"\n2. Loading input from {input_npy_path}")
    input_data = np.load(input_npy_path)
    input_tensor = torch.from_numpy(input_data).float()
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    elif input_tensor.dim() == 4:
        input_tensor = input_tensor.unsqueeze(0)
    print(f"  Input shape: {input_tensor.shape}")

    # Get MRI data for display (use preprocessed as background)
    mri_data = input_data[0] if input_data.ndim == 4 else input_data
    print(f"  MRI data shape: {mri_data.shape}")

    # Compute GradCAM
    print(f"\n3. Computing GradCAM++ on layer: {target_layer}")
    cam = compute_improved_gradcam(
        model, input_tensor,
        target_layer=target_layer,
        target_class=target_class,
        device=device,
        use_gradcam_pp=use_gradcam_pp
    )

    # Upsample and smooth
    print(f"\n4. Upsampling with order={interpolation_order}, sigma={sigma}")
    cam_smooth = upsample_and_smooth(
        cam, target_shape=mri_data.shape,
        interpolation_order=interpolation_order,
        sigma=sigma
    )
    print(f"  Smoothed CAM shape: {cam_smooth.shape}")

    # Brain masking
    print("\n5. Applying brain mask")
    # Auto-detect mask path from input npy path if not provided
    if mask_path is None:
        auto_mask = input_npy_path.replace('.npy', '_mask.npy')
        if os.path.exists(auto_mask):
            mask_path = auto_mask
            print(f"  Auto-detected mask: {mask_path}")
    brain_mask = create_brain_mask(mri_data, mask_path=mask_path)
    brain_pct = brain_mask.sum() / brain_mask.size * 100
    print(f"  Brain mask covers {brain_pct:.1f}% of volume")
    cam_masked = apply_brain_mask(cam_smooth, brain_mask)

    # Re-normalize after masking
    c_min, c_max = cam_masked.min(), cam_masked.max()
    if c_max - c_min > 1e-8:
        cam_masked = (cam_masked - c_min) / (c_max - c_min)

    # Save visualizations
    print(f"\n6. Saving visualizations to {output_dir}")
    png_paths = save_improved_slices(
        mri_data, cam_masked, output_dir,
        threshold_percentile=threshold_percentile,
        alpha=alpha,
        colormap=colormap,
        n_slices=n_slices
    )

    # Save attribution as npy for further analysis
    np.save(os.path.join(output_dir, 'attribution_improved.npy'), cam_masked)

    # Save config
    config = {
        'target_layer': target_layer,
        'target_class': target_class,
        'use_gradcam_pp': use_gradcam_pp,
        'interpolation_order': interpolation_order,
        'sigma': sigma,
        'threshold_percentile': threshold_percentile,
        'alpha': alpha,
        'colormap': colormap,
        'cam_raw_shape': list(cam.shape),
        'cam_smoothed_shape': list(cam_smooth.shape),
        'brain_mask_pct': float(brain_pct),
    }
    import json
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nDone! Generated {len(png_paths)} slices.")
    return cam_masked, png_paths


def main():
    parser = argparse.ArgumentParser(description='Improved MRI Heatmap Overlay')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--target_layer', type=str, default='bn3',
                        choices=['bn2', 'bn3', 'bn4'])
    parser.add_argument('--target_class', type=int, default=0)
    parser.add_argument('--no_gradcam_pp', action='store_true')
    parser.add_argument('--interpolation_order', type=int, default=3)
    parser.add_argument('--sigma', type=float, default=2.0)
    parser.add_argument('--threshold_percentile', type=float, default=85)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--colormap', type=str, default='hot_clinical')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_slices', type=int, default=5)
    parser.add_argument('--mask_path', type=str, default=None,
                        help='Path to pre-computed brain mask .npy (auto-detected if not provided)')
    args = parser.parse_args()

    run_improved_heatmap(
        checkpoint_path=args.checkpoint,
        input_npy_path=args.input,
        output_dir=args.output_dir,
        target_layer=args.target_layer,
        target_class=args.target_class,
        use_gradcam_pp=not args.no_gradcam_pp,
        interpolation_order=args.interpolation_order,
        sigma=args.sigma,
        threshold_percentile=args.threshold_percentile,
        alpha=args.alpha,
        colormap=args.colormap,
        device=args.device,
        n_slices=args.n_slices,
        mask_path=args.mask_path,
    )


if __name__ == '__main__':
    main()
