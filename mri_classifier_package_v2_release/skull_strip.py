#!/usr/bin/env python3
"""
Skull stripping utility using deepbet.

Wraps deepbet's brain extraction to provide two interfaces:
- skull_strip_nifti: operates on NIfTI file paths
- skull_strip_array: operates on numpy arrays (saves temp NIfTI, runs deepbet, loads back)

Usage:
    from skull_strip import skull_strip_nifti, skull_strip_array

    # From file
    brain_data, mask_data, affine = skull_strip_nifti('scan.nii.gz', '/tmp/output')

    # From array
    brain_data, mask_data = skull_strip_array(data, affine)
"""

import os
import tempfile
import numpy as np
import nibabel as nib
import warnings


def skull_strip_nifti(
    nifti_path: str,
    output_dir: str,
    threshold: float = 0.5,
    n_dilate: int = 0,
    no_gpu: bool = False
) -> tuple:
    """
    Skull strip a NIfTI file using deepbet.

    Args:
        nifti_path: Path to input .nii or .nii.gz file
        output_dir: Directory for output brain and mask files
        threshold: Brain mask threshold (0-1)
        n_dilate: Number of dilation iterations for mask
        no_gpu: If True, force CPU processing

    Returns:
        (brain_data, mask_data, affine) tuple
        - brain_data: numpy array with non-brain voxels zeroed
        - mask_data: boolean numpy array of brain mask
        - affine: 4x4 affine matrix from original NIfTI
    """
    from deepbet import run_bet

    os.makedirs(output_dir, exist_ok=True)

    basename = os.path.basename(nifti_path).replace('.nii.gz', '').replace('.nii', '')
    brain_path = os.path.join(output_dir, f'{basename}_brain.nii.gz')
    mask_path = os.path.join(output_dir, f'{basename}_mask.nii.gz')

    run_bet(
        input_paths=[nifti_path],
        brain_paths=[brain_path],
        mask_paths=[mask_path],
        threshold=threshold,
        n_dilate=n_dilate,
        no_gpu=no_gpu
    )

    # Load results
    brain_img = nib.load(brain_path)
    brain_data = brain_img.get_fdata()
    affine = brain_img.affine

    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata().astype(bool)

    return brain_data, mask_data, affine


def skull_strip_array(
    data: np.ndarray,
    affine: np.ndarray,
    threshold: float = 0.5,
    n_dilate: int = 0,
    no_gpu: bool = False
) -> tuple:
    """
    Skull strip a numpy array by saving to temp NIfTI, running deepbet, loading back.

    Args:
        data: 3D numpy array of MRI data
        affine: 4x4 affine matrix
        threshold: Brain mask threshold (0-1)
        n_dilate: Number of dilation iterations for mask
        no_gpu: If True, force CPU processing

    Returns:
        (brain_data, mask_data) tuple
        - brain_data: numpy array with non-brain voxels zeroed
        - mask_data: boolean numpy array of brain mask
    """
    from deepbet import run_bet

    with tempfile.TemporaryDirectory(prefix='skull_strip_') as tmpdir:
        # Save input as NIfTI
        input_path = os.path.join(tmpdir, 'input.nii.gz')
        brain_path = os.path.join(tmpdir, 'brain.nii.gz')
        mask_path = os.path.join(tmpdir, 'mask.nii.gz')

        img = nib.Nifti1Image(data.astype(np.float32), affine)
        nib.save(img, input_path)

        run_bet(
            input_paths=[input_path],
            brain_paths=[brain_path],
            mask_paths=[mask_path],
            threshold=threshold,
            n_dilate=n_dilate,
            no_gpu=no_gpu
        )

        # Load results
        brain_data = nib.load(brain_path).get_fdata()
        mask_data = nib.load(mask_path).get_fdata().astype(bool)

    return brain_data, mask_data


def skull_strip_array_safe(
    data: np.ndarray,
    affine: np.ndarray,
    **kwargs
) -> tuple:
    """
    Skull strip with fallback — returns original data if deepbet fails.

    Returns:
        (brain_data, mask_data, success) tuple
    """
    try:
        brain_data, mask_data = skull_strip_array(data, affine, **kwargs)
        return brain_data, mask_data, True
    except Exception as e:
        warnings.warn(f"Skull stripping failed, using original data: {e}")
        # Fallback: simple intensity threshold (assume mostly brain)
        mask_data = data > np.percentile(data[data > 0], 2) if (data > 0).any() else np.ones_like(data, dtype=bool)
        return data, mask_data.astype(bool), False
