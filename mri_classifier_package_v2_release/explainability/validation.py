"""
Validation utilities for MRI explainability pipeline.

Provides comprehensive validation at each stage of the pipeline to ensure
correct heatmap overlay on original brain images.
"""

import numpy as np
import warnings
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field


class ValidationError(Exception):
    """Raised when validation fails in strict mode."""
    pass


@dataclass
class ValidationResult:
    """Result of a validation check."""
    stage: str
    shape: Tuple
    checks: Dict[str, bool]
    all_passed: bool
    details: Dict[str, Any] = field(default_factory=dict)


class ExplainabilityValidator:
    """
    Comprehensive validation for explainability pipeline.

    Validates:
    1. Shape correctness at each step
    2. Value range constraints
    3. NaN/Inf detection
    4. Affine alignment
    5. Spatial consistency

    Args:
        strict_mode: If True, raise ValidationError on failures.
                    If False, only warn.
    """

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.validation_log = []

    def check_input_attribution(self, attr: np.ndarray) -> ValidationResult:
        """
        Validate input attribution from explainer.

        Expected: (B, C, D, H, W) or (C, D, H, W) or (D, H, W)
        """
        checks = {
            'shape_valid': attr.ndim in [3, 4, 5],
            'no_nan': not np.isnan(attr).any(),
            'no_inf': not np.isinf(attr).any(),
            'has_variance': attr.std() > 1e-10,
            'non_empty': attr.size > 0,
        }

        details = {
            'ndim': attr.ndim,
            'dtype': str(attr.dtype),
            'min': float(attr.min()) if attr.size > 0 else None,
            'max': float(attr.max()) if attr.size > 0 else None,
            'std': float(attr.std()) if attr.size > 0 else None,
            'nan_count': int(np.isnan(attr).sum()),
            'inf_count': int(np.isinf(attr).sum()),
        }

        return self._log_and_raise('input_attribution', checks, attr.shape, details)

    def check_after_squeeze(
        self,
        attr: np.ndarray,
        expected_shape: Tuple
    ) -> ValidationResult:
        """
        Validate after removing batch/channel dims.

        Expected: 3D array (D, H, W)
        """
        checks = {
            'is_3d': attr.ndim == 3,
            'shape_match': attr.shape == expected_shape,
            'no_nan': not np.isnan(attr).any(),
            'no_inf': not np.isinf(attr).any(),
        }

        details = {
            'actual_shape': attr.shape,
            'expected_shape': expected_shape,
        }

        return self._log_and_raise('after_squeeze', checks, attr.shape, details)

    def check_after_upsample_to_preprocessed(
        self,
        attr: np.ndarray,
        expected_shape: Tuple = (128, 128, 128)
    ) -> ValidationResult:
        """
        Validate after upsampling to preprocessed resolution (128x128x128).
        """
        checks = {
            'shape_match': attr.shape == expected_shape,
            'no_nan': not np.isnan(attr).any(),
            'no_inf': not np.isinf(attr).any(),
            'non_negative': attr.min() >= -1e-6,  # Small tolerance for floating point
        }

        details = {
            'actual_shape': attr.shape,
            'expected_shape': expected_shape,
            'min': float(attr.min()),
            'max': float(attr.max()),
        }

        return self._log_and_raise('upsample_to_preprocessed', checks, attr.shape, details)

    def check_after_upsample_to_original(
        self,
        attr: np.ndarray,
        expected_shape: Tuple
    ) -> ValidationResult:
        """
        Validate after upsampling to original NIfTI dimensions.
        """
        # Allow small tolerance in shape matching due to rounding
        shape_close = all(
            abs(a - e) <= 1 for a, e in zip(attr.shape, expected_shape)
        )

        checks = {
            'shape_match': attr.shape == expected_shape,
            'shape_close': shape_close,  # Within 1 voxel tolerance
            'no_nan': not np.isnan(attr).any(),
            'no_inf': not np.isinf(attr).any(),
            'non_negative': attr.min() >= -1e-6,
        }

        details = {
            'actual_shape': attr.shape,
            'expected_shape': expected_shape,
            'shape_diff': tuple(a - e for a, e in zip(attr.shape, expected_shape)),
        }

        return self._log_and_raise('upsample_to_original', checks, attr.shape, details)

    def check_final_attribution(
        self,
        attr: np.ndarray,
        expected_shape: Tuple,
        affine: np.ndarray
    ) -> ValidationResult:
        """
        Final validation before output.

        Checks:
        - Shape matches original NIfTI
        - Values normalized to [0, 1]
        - Affine is valid and invertible
        """
        checks = {
            'shape_match': attr.shape == expected_shape,
            'normalized_min': attr.min() >= -1e-6,
            'normalized_max': attr.max() <= 1.0 + 1e-6,
            'no_nan': not np.isnan(attr).any(),
            'no_inf': not np.isinf(attr).any(),
            'affine_shape': affine.shape == (4, 4),
            'affine_invertible': abs(np.linalg.det(affine[:3, :3])) > 1e-10,
        }

        details = {
            'actual_shape': attr.shape,
            'expected_shape': expected_shape,
            'value_range': (float(attr.min()), float(attr.max())),
            'affine_det': float(np.linalg.det(affine[:3, :3])),
        }

        return self._log_and_raise('final_attribution', checks, attr.shape, details)

    def check_nifti_alignment(
        self,
        original_shape: Tuple,
        original_affine: np.ndarray,
        attribution_shape: Tuple,
        attribution_affine: np.ndarray
    ) -> ValidationResult:
        """
        Verify spatial alignment between original and attribution.
        """
        import nibabel as nib

        affine_close = np.allclose(original_affine, attribution_affine, rtol=1e-5)

        # Get axis orientations
        orig_orient = nib.aff2axcodes(original_affine)
        attr_orient = nib.aff2axcodes(attribution_affine)

        checks = {
            'shape_match': original_shape[:3] == attribution_shape[:3],
            'affine_close': affine_close,
            'same_orientation': orig_orient == attr_orient,
        }

        details = {
            'original_shape': original_shape,
            'attribution_shape': attribution_shape,
            'original_orientation': orig_orient,
            'attribution_orientation': attr_orient,
            'affine_max_diff': float(np.abs(original_affine - attribution_affine).max()),
        }

        return self._log_and_raise('nifti_alignment', checks, attribution_shape, details)

    def check_coverage(
        self,
        attribution: np.ndarray,
        threshold: float = 0.1,
        min_coverage: float = 0.01,
        max_coverage: float = 0.99
    ) -> ValidationResult:
        """
        Check that attribution has reasonable coverage.

        - Not all zeros (no signal)
        - Not all ones (no specificity)
        """
        coverage = (attribution > threshold).sum() / attribution.size

        checks = {
            'not_all_zeros': coverage > min_coverage,
            'not_all_ones': coverage < max_coverage,
            'reasonable_coverage': min_coverage < coverage < max_coverage,
        }

        details = {
            'coverage': float(coverage),
            'threshold': threshold,
            'min_coverage': min_coverage,
            'max_coverage': max_coverage,
            'num_above_threshold': int((attribution > threshold).sum()),
            'total_voxels': attribution.size,
        }

        return self._log_and_raise('coverage', checks, attribution.shape, details)

    def check_brain_overlap(
        self,
        attribution: np.ndarray,
        original_data: np.ndarray,
        min_overlap: float = 0.5
    ) -> ValidationResult:
        """
        Check that high attribution regions overlap with brain.

        Uses simple intensity thresholding as brain mask proxy.
        """
        # Create simple brain mask based on intensity
        brain_threshold = original_data.mean()
        brain_mask = original_data > brain_threshold

        # Check overlap
        attr_mask = attribution > 0.1
        if attr_mask.sum() > 0:
            overlap = (attr_mask & brain_mask).sum() / attr_mask.sum()
        else:
            overlap = 0.0

        checks = {
            'sufficient_overlap': overlap >= min_overlap,
            'has_attribution': attr_mask.sum() > 0,
        }

        details = {
            'overlap_ratio': float(overlap),
            'min_overlap': min_overlap,
            'brain_voxels': int(brain_mask.sum()),
            'attribution_voxels': int(attr_mask.sum()),
            'overlapping_voxels': int((attr_mask & brain_mask).sum()),
        }

        return self._log_and_raise('brain_overlap', checks, attribution.shape, details)

    def check_smoothness(
        self,
        attribution: np.ndarray,
        max_gradient: float = 0.5
    ) -> ValidationResult:
        """
        Check spatial smoothness of attribution.

        Very sharp discontinuities may indicate interpolation artifacts.
        """
        # Compute gradient magnitude
        gradients = np.gradient(attribution)
        gradient_magnitude = np.sqrt(sum(g**2 for g in gradients))
        max_grad = gradient_magnitude.max()
        mean_grad = gradient_magnitude.mean()

        checks = {
            'no_extreme_gradients': max_grad <= max_gradient,
            'smooth_overall': mean_grad < max_gradient / 10,
        }

        details = {
            'max_gradient': float(max_grad),
            'mean_gradient': float(mean_grad),
            'threshold': max_gradient,
        }

        # Only warn for smoothness issues, don't fail
        result = ValidationResult(
            stage='smoothness',
            shape=attribution.shape,
            checks=checks,
            all_passed=all(checks.values()),
            details=details
        )
        self.validation_log.append(result)

        if not result.all_passed:
            failed = [k for k, v in checks.items() if not v]
            warnings.warn(f"Smoothness check: {failed}. Max gradient: {max_grad:.3f}")

        return result

    def _log_and_raise(
        self,
        stage: str,
        checks: Dict[str, bool],
        shape: Tuple,
        details: Dict[str, Any] = None
    ) -> ValidationResult:
        """Log validation results and raise if failures in strict mode."""
        result = ValidationResult(
            stage=stage,
            shape=shape,
            checks=checks,
            all_passed=all(checks.values()),
            details=details or {}
        )
        self.validation_log.append(result)

        if not result.all_passed:
            failed = [k for k, v in checks.items() if not v]
            msg = f"Validation failed at {stage}: {failed}. Shape: {shape}. Details: {details}"

            if self.strict_mode:
                raise ValidationError(msg)
            else:
                warnings.warn(msg)

        return result

    def get_validation_report(self) -> Dict[str, Any]:
        """Get full validation report as dictionary."""
        return {
            'stages': [
                {
                    'stage': r.stage,
                    'shape': r.shape,
                    'checks': r.checks,
                    'all_passed': r.all_passed,
                    'details': r.details
                }
                for r in self.validation_log
            ],
            'all_passed': all(r.all_passed for r in self.validation_log),
            'total_stages': len(self.validation_log),
            'failed_stages': [r.stage for r in self.validation_log if not r.all_passed]
        }

    def reset(self):
        """Clear validation log for new run."""
        self.validation_log = []


def run_all_sanity_checks(
    attribution: np.ndarray,
    original_data: np.ndarray,
    original_shape: Tuple,
    affine: np.ndarray,
    strict: bool = False
) -> Dict[str, Any]:
    """
    Run all sanity checks on final attribution.

    Convenience function that runs coverage, brain overlap, and smoothness checks.

    Args:
        attribution: Final attribution array in original space
        original_data: Original MRI data
        original_shape: Expected shape
        affine: Affine matrix
        strict: Whether to raise on failures

    Returns:
        Dictionary with all check results
    """
    validator = ExplainabilityValidator(strict_mode=strict)

    # Run all checks
    validator.check_final_attribution(attribution, original_shape, affine)
    validator.check_coverage(attribution)
    validator.check_brain_overlap(attribution, original_data)
    validator.check_smoothness(attribution)

    return validator.get_validation_report()
