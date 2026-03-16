"""
Reverse Transform Pipeline for MRI Explainability.

Transforms attribution heatmaps from model space back to original NIfTI space.
"""

import numpy as np
import torch
from scipy.ndimage import zoom
from typing import Tuple, Optional
from .validation import ExplainabilityValidator


class ReverseTransformer:
    """
    Reverse transform heatmaps from preprocessed space to original NIfTI space.

    Preprocessing pipeline (forward) that this reverses:
    1. Original NIfTI: (D_orig, H_orig, W_orig) variable shape
    2. Resize: scipy.ndimage.zoom with order=1 to (128, 128, 128)
    3. Normalize: min-max to [0, 1]
    4. Add channel: (1, 128, 128, 128)

    Reverse pipeline:
    1. Remove channel dim: (128, 128, 128) or (8, 8, 8)
    2. Upsample heatmap to 128x128x128 if from GradCAM
    3. Upsample to original spatial dimensions
    4. Normalize to [0, 1]
    """

    PREPROCESSED_SHAPE = (128, 128, 128)

    def __init__(self, interpolation_order: int = 1):
        """
        Args:
            interpolation_order: Order of spline interpolation for scipy.ndimage.zoom
                                0 = nearest neighbor
                                1 = trilinear (recommended)
                                3 = cubic
        """
        self.interpolation_order = interpolation_order

    def transform(
        self,
        attribution: torch.Tensor,
        original_shape: Tuple[int, int, int],
        affine: np.ndarray,
        validator: Optional[ExplainabilityValidator] = None
    ) -> np.ndarray:
        """
        Transform attribution map to original NIfTI space.

        Args:
            attribution: (B, C, D, H, W) or (C, D, H, W) or (D, H, W) tensor
            original_shape: (D_orig, H_orig, W_orig) from original NIfTI
            affine: 4x4 affine matrix from original NIfTI
            validator: Optional validator for checks at each step

        Returns:
            np.ndarray: (D_orig, H_orig, W_orig) attribution in original space
        """
        # Step 1: Convert to numpy and validate input
        attr_np = self._to_numpy(attribution)

        if validator:
            validator.check_input_attribution(attr_np)

        # Step 2: Remove batch and channel dimensions
        attr_np = self._squeeze_dims(attr_np)
        current_shape = attr_np.shape

        if validator:
            validator.check_after_squeeze(attr_np, current_shape)

        # Step 3: Upsample to preprocessed resolution if needed
        if current_shape != self.PREPROCESSED_SHAPE:
            attr_np = self._upsample_to_preprocessed(attr_np)

            if validator:
                validator.check_after_upsample_to_preprocessed(
                    attr_np, self.PREPROCESSED_SHAPE
                )

        # Step 4: Upsample to original spatial dimensions
        attr_np = self._upsample_to_original(attr_np, original_shape)

        if validator:
            validator.check_after_upsample_to_original(attr_np, original_shape)

        # Step 5: Normalize to [0, 1] range
        attr_np = self._normalize(attr_np)

        if validator:
            validator.check_final_attribution(attr_np, original_shape, affine)

        return attr_np

    def _to_numpy(self, attribution: torch.Tensor) -> np.ndarray:
        """Convert attribution to numpy array."""
        if isinstance(attribution, torch.Tensor):
            return attribution.detach().cpu().numpy()
        return np.array(attribution)

    def _squeeze_dims(self, attr: np.ndarray) -> np.ndarray:
        """Remove batch and channel dimensions."""
        # (B, C, D, H, W) -> (D, H, W)
        if attr.ndim == 5:
            attr = attr[0, 0]
        # (C, D, H, W) -> (D, H, W)
        elif attr.ndim == 4:
            attr = attr[0]
        # (D, H, W) stays as is

        return attr

    def _upsample_to_preprocessed(self, attr: np.ndarray) -> np.ndarray:
        """
        Upsample attribution to preprocessed resolution (128, 128, 128).

        Typically used for GradCAM output which is (8, 8, 8).
        """
        current_shape = attr.shape
        zoom_factors = tuple(
            p / c for p, c in zip(self.PREPROCESSED_SHAPE, current_shape)
        )

        return zoom(attr, zoom_factors, order=self.interpolation_order)

    def _upsample_to_original(
        self,
        attr: np.ndarray,
        original_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Upsample attribution to original NIfTI dimensions.
        """
        zoom_factors = tuple(
            o / p for o, p in zip(original_shape, self.PREPROCESSED_SHAPE)
        )

        return zoom(attr, zoom_factors, order=self.interpolation_order)

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize to [0, 1] range."""
        d_min, d_max = data.min(), data.max()
        if d_max - d_min > 1e-8:
            return (data - d_min) / (d_max - d_min)
        return np.zeros_like(data)


class ReverseTransformPipeline:
    """
    Complete reverse transform pipeline with validation.

    Wraps ReverseTransformer with built-in validation and sanity checks.
    """

    def __init__(
        self,
        strict_validation: bool = True,
        interpolation_order: int = 1
    ):
        self.transformer = ReverseTransformer(interpolation_order)
        self.validator = ExplainabilityValidator(strict_mode=strict_validation)

    def transform(
        self,
        attribution: torch.Tensor,
        original_shape: Tuple[int, int, int],
        affine: np.ndarray,
        original_data: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Transform attribution with full validation.

        Args:
            attribution: Attribution tensor from explainer
            original_shape: Shape of original NIfTI
            affine: Affine matrix from original NIfTI
            original_data: Original MRI data for brain overlap check

        Returns:
            Tuple of (transformed attribution, validation report)
        """
        # Reset validator for fresh run
        self.validator.reset()

        # Run transform with validation
        attr_transformed = self.transformer.transform(
            attribution,
            original_shape,
            affine,
            validator=self.validator
        )

        # Additional sanity checks if original data provided
        if original_data is not None:
            self.validator.check_coverage(attr_transformed)
            self.validator.check_brain_overlap(attr_transformed, original_data)
            self.validator.check_smoothness(attr_transformed)

        return attr_transformed, self.validator.get_validation_report()


def compute_zoom_factors(
    source_shape: Tuple[int, int, int],
    target_shape: Tuple[int, int, int]
) -> Tuple[float, float, float]:
    """
    Compute zoom factors to transform from source to target shape.

    Args:
        source_shape: Current shape (D, H, W)
        target_shape: Desired shape (D, H, W)

    Returns:
        Tuple of zoom factors (zf_d, zf_h, zf_w)
    """
    return tuple(t / s for t, s in zip(target_shape, source_shape))


def validate_transform_reversibility(
    original_shape: Tuple[int, int, int],
    preprocessed_shape: Tuple[int, int, int] = (128, 128, 128),
    tolerance: float = 0.01
) -> dict:
    """
    Validate that transform is reversible without significant information loss.

    Creates synthetic data, transforms forward and back, measures reconstruction error.
    """
    # Create synthetic data with some structure
    np.random.seed(42)
    original = np.random.randn(*original_shape).astype(np.float32)
    original = zoom(original, (0.5, 0.5, 0.5), order=1)  # Smooth it
    original = zoom(original, (2, 2, 2), order=1)

    # Forward transform (simulate preprocessing)
    forward_zoom = tuple(p / o for p, o in zip(preprocessed_shape, original_shape))
    preprocessed = zoom(original, forward_zoom, order=1)

    # Reverse transform
    reverse_zoom = tuple(o / p for o, p in zip(original_shape, preprocessed_shape))
    reconstructed = zoom(preprocessed, reverse_zoom, order=1)

    # Compute error
    mse = np.mean((original - reconstructed) ** 2)
    max_error = np.max(np.abs(original - reconstructed))
    correlation = np.corrcoef(original.flatten(), reconstructed.flatten())[0, 1]

    return {
        'mse': float(mse),
        'max_error': float(max_error),
        'correlation': float(correlation),
        'reversible': max_error < tolerance * (original.max() - original.min()),
        'original_shape': original_shape,
        'preprocessed_shape': preprocessed_shape,
    }
