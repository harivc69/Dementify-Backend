"""
Visualization Module for MRI Explainability.

Generates NIfTI files and PNG slices for viewing attribution maps.
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import json


@dataclass
class VisualizationResult:
    """Result from visualization generation."""
    nifti_path: Optional[str]
    overlay_path: Optional[str]
    png_paths: List[str]
    validation_report_path: Optional[str]


class ExplainabilityVisualizer:
    """
    Generate visualization outputs for explainability results.

    Outputs:
    1. NIfTI files for 3D viewers (ITK-SNAP, FSLeyes, 3D Slicer)
    2. PNG slices (axial, coronal, sagittal) with overlay
    3. Validation report JSON
    """

    def __init__(
        self,
        colormap: str = 'hot',
        overlay_alpha: float = 0.6,
        threshold: float = 0.2
    ):
        """
        Initialize visualizer.

        Args:
            colormap: Colormap for attribution ('hot', 'jet', 'inferno')
            overlay_alpha: Transparency for overlay (0-1)
            threshold: Threshold for masking low attribution values
        """
        self.colormap = colormap
        self.overlay_alpha = overlay_alpha
        self.threshold = threshold

    def save_all(
        self,
        attribution: np.ndarray,
        affine: np.ndarray,
        output_dir: str,
        original_nifti: Optional[nib.Nifti1Image] = None,
        original_data: Optional[np.ndarray] = None,
        validation_report: Optional[Dict] = None,
        n_slices: int = 5
    ) -> VisualizationResult:
        """
        Save all visualization outputs.

        Args:
            attribution: Attribution map in original space (D, H, W)
            affine: Affine matrix from original NIfTI
            output_dir: Directory to save outputs
            original_nifti: Original NIfTI image for overlay
            original_data: Raw 3D numpy array as fallback background
            validation_report: Validation report to save
            n_slices: Number of PNG slices per view

        Returns:
            VisualizationResult with paths to all outputs
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save attribution NIfTI
        nifti_path = os.path.join(output_dir, 'attribution.nii.gz')
        self.save_nifti(attribution, affine, nifti_path, original_nifti)

        # Save overlay NIfTI if original provided
        overlay_path = None
        if original_nifti is not None:
            overlay_path = os.path.join(output_dir, 'attribution_overlay.nii.gz')
            self.save_overlay_nifti(original_nifti, attribution, overlay_path)

        # Save PNG slices
        png_paths = []
        if original_nifti is not None:
            png_dir = output_dir
            png_paths = self.save_png_slices(
                original_nifti, attribution, png_dir, n_slices
            )
        elif original_data is not None:
            png_dir = output_dir
            png_paths = self.save_png_slices_from_array(
                original_data, attribution, png_dir, n_slices
            )

        # Save validation report
        report_path = None
        if validation_report is not None:
            report_path = os.path.join(output_dir, 'validation_report.json')
            with open(report_path, 'w') as f:
                json.dump(validation_report, f, indent=2, default=str)

        return VisualizationResult(
            nifti_path=nifti_path,
            overlay_path=overlay_path,
            png_paths=png_paths,
            validation_report_path=report_path
        )

    def save_nifti(
        self,
        attribution: np.ndarray,
        affine: np.ndarray,
        output_path: str,
        original_nifti: Optional[nib.Nifti1Image] = None
    ):
        """
        Save attribution as NIfTI file.

        Preserves original affine for proper spatial alignment.
        """
        # Scale to 0-1000 for better visualization in viewers
        # (Many viewers expect non-normalized intensities)
        attr_scaled = (attribution * 1000).astype(np.float32)

        nifti_img = nib.Nifti1Image(attr_scaled, affine)

        # Copy header info from original if available
        if original_nifti is not None:
            units = original_nifti.header.get_xyzt_units()
            nifti_img.header.set_xyzt_units(units[0], units[1])

        nib.save(nifti_img, output_path)
        print(f"Saved attribution NIfTI: {output_path}")

    def save_overlay_nifti(
        self,
        original_nifti: nib.Nifti1Image,
        attribution: np.ndarray,
        output_path: str
    ):
        """
        Save as overlay-ready NIfTI (4D with original + heatmap).

        Can be loaded in FSLeyes/ITK-SNAP as dual overlay.
        """
        original_data = original_nifti.get_fdata()

        # Handle 4D original (take first volume)
        if original_data.ndim == 4:
            original_data = original_data[..., 0]

        # Normalize original to [0, 1]
        orig_min, orig_max = original_data.min(), original_data.max()
        if orig_max > orig_min:
            orig_norm = (original_data - orig_min) / (orig_max - orig_min)
        else:
            orig_norm = np.zeros_like(original_data)

        # Stack as 4D: [original, attribution]
        combined = np.stack([orig_norm, attribution], axis=-1).astype(np.float32)

        overlay_nifti = nib.Nifti1Image(combined, original_nifti.affine)
        nib.save(overlay_nifti, output_path)
        print(f"Saved overlay NIfTI: {output_path}")

    def save_png_slices(
        self,
        original_nifti: nib.Nifti1Image,
        attribution: np.ndarray,
        output_dir: str,
        n_slices: int = 5
    ) -> List[str]:
        """
        Save PNG slices with attribution overlay.

        Generates axial, coronal, and sagittal views.
        """
        original_data = original_nifti.get_fdata()

        # Handle 4D
        if original_data.ndim == 4:
            original_data = original_data[..., 0]

        # Normalize original for display
        orig_min, orig_max = original_data.min(), original_data.max()
        if orig_max > orig_min:
            orig_display = (original_data - orig_min) / (orig_max - orig_min)
        else:
            orig_display = np.zeros_like(original_data)

        views = {
            'axial': (2, 'z'),      # Along z-axis
            'coronal': (1, 'y'),    # Along y-axis
            'sagittal': (0, 'x'),   # Along x-axis
        }

        png_paths = []

        for view_name, (axis, label) in views.items():
            view_dir = os.path.join(output_dir, view_name)
            os.makedirs(view_dir, exist_ok=True)

            # Select slice indices (avoid edges)
            n_total = original_data.shape[axis]
            start = n_total // 5
            end = n_total * 4 // 5
            indices = np.linspace(start, end, n_slices, dtype=int)

            for i, idx in enumerate(indices):
                # Extract slices
                orig_slice = np.take(orig_display, idx, axis=axis)
                attr_slice = np.take(attribution, idx, axis=axis)

                # Create figure with 3 panels
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # Rotate for proper orientation
                orig_slice = np.rot90(orig_slice)
                attr_slice = np.rot90(attr_slice)

                # Panel 1: Original
                axes[0].imshow(orig_slice, cmap='gray', vmin=0, vmax=1)
                axes[0].set_title(f'Original ({label}={idx})', fontsize=12)
                axes[0].axis('off')

                # Panel 2: Attribution
                im = axes[1].imshow(attr_slice, cmap=self.colormap, vmin=0, vmax=1)
                axes[1].set_title('Attribution', fontsize=12)
                axes[1].axis('off')
                plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

                # Panel 3: Overlay
                axes[2].imshow(orig_slice, cmap='gray', vmin=0, vmax=1)
                # Mask low values for cleaner overlay
                masked_attr = np.ma.masked_where(
                    attr_slice < self.threshold, attr_slice
                )
                axes[2].imshow(
                    masked_attr, cmap=self.colormap,
                    alpha=self.overlay_alpha, vmin=0, vmax=1
                )
                axes[2].set_title(f'Overlay (threshold={self.threshold})', fontsize=12)
                axes[2].axis('off')

                plt.tight_layout()

                # Save
                png_path = os.path.join(view_dir, f'slice_{i:02d}_{label}{idx}.png')
                plt.savefig(png_path, dpi=150, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                plt.close()

                png_paths.append(png_path)

        print(f"Saved {len(png_paths)} PNG slices to {output_dir}")
        return png_paths


    def save_png_slices_from_array(
        self,
        original_data: np.ndarray,
        attribution: np.ndarray,
        output_dir: str,
        n_slices: int = 5
    ) -> List[str]:
        """Save PNG slices using a raw numpy array as background."""
        if original_data.ndim == 4:
            original_data = original_data[0]

        orig_min, orig_max = original_data.min(), original_data.max()
        if orig_max > orig_min:
            orig_display = (original_data - orig_min) / (orig_max - orig_min)
        else:
            orig_display = np.zeros_like(original_data)

        views = {
            'axial': (2, 'z'),
            'coronal': (1, 'y'),
            'sagittal': (0, 'x'),
        }

        png_paths = []
        for view_name, (axis, label) in views.items():
            view_dir = os.path.join(output_dir, view_name)
            os.makedirs(view_dir, exist_ok=True)

            n_total = original_data.shape[axis]
            start = n_total // 5
            end = n_total * 4 // 5
            indices = np.linspace(start, end, n_slices, dtype=int)

            for i, idx in enumerate(indices):
                orig_slice = np.rot90(np.take(orig_display, idx, axis=axis))
                attr_slice = np.rot90(np.take(attribution, idx, axis=axis))

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                axes[0].imshow(orig_slice, cmap='gray', vmin=0, vmax=1)
                axes[0].set_title(f'Original ({label}={idx})', fontsize=12)
                axes[0].axis('off')

                im = axes[1].imshow(attr_slice, cmap=self.colormap, vmin=0, vmax=1)
                axes[1].set_title('Attribution', fontsize=12)
                axes[1].axis('off')
                plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

                axes[2].imshow(orig_slice, cmap='gray', vmin=0, vmax=1)
                masked_attr = np.ma.masked_where(
                    attr_slice < self.threshold, attr_slice
                )
                axes[2].imshow(
                    masked_attr, cmap=self.colormap,
                    alpha=self.overlay_alpha, vmin=0, vmax=1
                )
                axes[2].set_title(f'Overlay (threshold={self.threshold})', fontsize=12)
                axes[2].axis('off')

                plt.tight_layout()
                png_path = os.path.join(view_dir, f'slice_{i:02d}_{label}{idx}.png')
                plt.savefig(png_path, dpi=150, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                plt.close()
                png_paths.append(png_path)

        print(f"Saved {len(png_paths)} PNG slices to {output_dir}")
        return png_paths


def create_comparison_figure(
    original_data: np.ndarray,
    gradcam_attr: np.ndarray,
    shap_attr: np.ndarray,
    slice_idx: int,
    axis: int = 2,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comparison figure between GradCAM and SHAP attributions.

    Args:
        original_data: Original MRI data
        gradcam_attr: GradCAM attribution
        shap_attr: SHAP attribution
        slice_idx: Slice index to visualize
        axis: Axis to slice along (0=x, 1=y, 2=z)
        output_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    # Extract slices
    orig_slice = np.rot90(np.take(original_data, slice_idx, axis=axis))
    gc_slice = np.rot90(np.take(gradcam_attr, slice_idx, axis=axis))
    shap_slice = np.rot90(np.take(shap_attr, slice_idx, axis=axis))

    # Normalize
    orig_slice = (orig_slice - orig_slice.min()) / (orig_slice.max() - orig_slice.min() + 1e-8)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: GradCAM
    axes[0, 0].imshow(orig_slice, cmap='gray')
    axes[0, 0].set_title('Original MRI', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gc_slice, cmap='hot', vmin=0, vmax=1)
    axes[0, 1].set_title('GradCAM Attribution', fontsize=12)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(orig_slice, cmap='gray')
    masked_gc = np.ma.masked_where(gc_slice < 0.2, gc_slice)
    axes[0, 2].imshow(masked_gc, cmap='hot', alpha=0.6, vmin=0, vmax=1)
    axes[0, 2].set_title('GradCAM Overlay', fontsize=12)
    axes[0, 2].axis('off')

    # Row 2: SHAP
    axes[1, 0].imshow(orig_slice, cmap='gray')
    axes[1, 0].set_title('Original MRI', fontsize=12)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(shap_slice, cmap='hot', vmin=0, vmax=1)
    axes[1, 1].set_title('SHAP Attribution', fontsize=12)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(orig_slice, cmap='gray')
    masked_shap = np.ma.masked_where(shap_slice < 0.2, shap_slice)
    axes[1, 2].imshow(masked_shap, cmap='hot', alpha=0.6, vmin=0, vmax=1)
    axes[1, 2].set_title('SHAP Overlay', fontsize=12)
    axes[1, 2].axis('off')

    plt.suptitle(f'GradCAM vs SHAP Comparison (Slice {slice_idx})', fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Saved comparison figure: {output_path}")

    return fig


def create_summary_mosaic(
    original_data: np.ndarray,
    attribution: np.ndarray,
    output_path: str,
    n_rows: int = 4,
    n_cols: int = 5,
    threshold: float = 0.2
):
    """
    Create a mosaic summary of all slices with attribution overlay.

    Args:
        original_data: Original MRI data (D, H, W)
        attribution: Attribution map (D, H, W)
        output_path: Path to save mosaic
        n_rows: Number of rows in mosaic
        n_cols: Number of columns in mosaic
        threshold: Attribution threshold for overlay
    """
    # Normalize original
    orig_norm = (original_data - original_data.min()) / \
                (original_data.max() - original_data.min() + 1e-8)

    # Select slices (axial view)
    n_slices = n_rows * n_cols
    total_slices = original_data.shape[2]
    indices = np.linspace(
        total_slices // 10,
        total_slices * 9 // 10,
        n_slices,
        dtype=int
    )

    # Create mosaic
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for i, (ax, idx) in enumerate(zip(axes, indices)):
        orig_slice = np.rot90(orig_norm[:, :, idx])
        attr_slice = np.rot90(attribution[:, :, idx])

        ax.imshow(orig_slice, cmap='gray')
        masked_attr = np.ma.masked_where(attr_slice < threshold, attr_slice)
        ax.imshow(masked_attr, cmap='hot', alpha=0.6, vmin=0, vmax=1)
        ax.set_title(f'z={idx}', fontsize=8)
        ax.axis('off')

    plt.suptitle('Attribution Mosaic (Axial Slices)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved mosaic: {output_path}")
