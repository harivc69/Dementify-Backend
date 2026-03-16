"""
SHAP Explainer for LightMRI3D Model.

Uses supervoxel-based approach for computational efficiency on 3D MRI data.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from scipy.ndimage import zoom
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP library not installed. Run: pip install shap")


@dataclass
class SHAPResult:
    """Result from SHAP computation."""
    shap_values: np.ndarray  # (B, 1, D, H, W) SHAP values
    base_value: float  # Expected model output
    prediction: float  # Model prediction for this input
    method: str  # 'supervoxel' or 'voxel'
    supervoxel_size: Optional[int] = None


class SHAPExplainerMRI:
    """
    SHAP Explainer for MRI using supervoxel optimization.

    Challenge: 128^3 = 2M voxels is too expensive for direct SHAP

    Solution: Supervoxel-based SHAP
    - Divide volume into supervoxels (e.g., 8x8x8 regions)
    - Compute SHAP values for each supervoxel
    - Upsample to full resolution

    With 8x8x8 supervoxels: 128/8 = 16 supervoxels per dimension
    Total: 16^3 = 4096 supervoxels (much more tractable)
    """

    def __init__(
        self,
        model: nn.Module,
        background_data: Optional[torch.Tensor] = None,
        n_background_samples: int = 10,
        supervoxel_size: int = 8,
        device: str = 'cuda'
    ):
        """
        Initialize SHAP explainer.

        Args:
            model: LightMRI3D model (or wrapper)
            background_data: Background samples for SHAP (B, 1, 128, 128, 128)
            n_background_samples: Number of background samples to use
            supervoxel_size: Size of supervoxels (8 = 8x8x8 regions)
            device: Device to run on
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library required. Install with: pip install shap")

        self.device = device
        self.supervoxel_size = supervoxel_size
        self.n_background_samples = n_background_samples

        # Extract MRI model if wrapped
        self.model = self._extract_mri_model(model)
        self.model.eval()

        # Setup background data
        self.background_data = background_data
        self._setup_background()

        # Compute supervoxel dimensions
        self.input_shape = (128, 128, 128)
        self.sv_dims = tuple(d // supervoxel_size for d in self.input_shape)
        self.n_supervoxels = np.prod(self.sv_dims)

        print(f"SHAP Explainer initialized:")
        print(f"  Supervoxel size: {supervoxel_size}")
        print(f"  Supervoxel dimensions: {self.sv_dims}")
        print(f"  Total supervoxels: {self.n_supervoxels}")

    def _extract_mri_model(self, model: nn.Module) -> nn.Module:
        """Extract LightMRI3D from wrapper if needed."""
        if hasattr(model, 'img_model'):
            return model.img_model.to(self.device)

        if hasattr(model, 'modules_emb_src'):
            for key, module in model.modules_emb_src.items():
                if hasattr(module, 'img_model'):
                    return module.img_model.to(self.device)

        return model.to(self.device)

    def _setup_background(self):
        """Setup background data for SHAP."""
        if self.background_data is None:
            # Create default background (zeros)
            self.background_data = torch.zeros(
                self.n_background_samples, 1, 128, 128, 128
            )
            print("Using zero background (no background data provided)")
        else:
            # Subsample if needed
            if len(self.background_data) > self.n_background_samples:
                indices = np.random.choice(
                    len(self.background_data),
                    self.n_background_samples,
                    replace=False
                )
                self.background_data = self.background_data[indices]

        self.background_data = self.background_data.to(self.device)

    def compute(
        self,
        input_tensor: torch.Tensor,
        n_samples: int = 100
    ) -> SHAPResult:
        """
        Compute SHAP values using supervoxel masking.

        Args:
            input_tensor: Input MRI (B, 1, 128, 128, 128)
            n_samples: Number of samples for SHAP estimation

        Returns:
            SHAPResult with SHAP values and metadata
        """
        input_tensor = input_tensor.to(self.device)

        # Get baseline prediction (with all supervoxels)
        with torch.no_grad():
            baseline_output = self.model(input_tensor)
            if baseline_output.dim() > 1:
                baseline_pred = baseline_output.mean().item()
            else:
                baseline_pred = baseline_output.item()

        # Get expected value (background prediction)
        with torch.no_grad():
            bg_outputs = self.model(self.background_data)
            if bg_outputs.dim() > 1:
                base_value = bg_outputs.mean().item()
            else:
                base_value = bg_outputs.mean().item()

        # Compute supervoxel SHAP values
        sv_shap_values = self._compute_supervoxel_shap(
            input_tensor, n_samples
        )

        # Upsample to full resolution
        full_shap_values = self._upsample_supervoxels(sv_shap_values)

        return SHAPResult(
            shap_values=full_shap_values,
            base_value=base_value,
            prediction=baseline_pred,
            method='supervoxel',
            supervoxel_size=self.supervoxel_size
        )

    def _compute_supervoxel_shap(
        self,
        input_tensor: torch.Tensor,
        n_samples: int
    ) -> np.ndarray:
        """
        Compute SHAP values at supervoxel level.

        Uses permutation-based sampling to estimate Shapley values.
        """
        input_np = input_tensor.cpu().numpy()
        sv_shape = self.sv_dims

        # Initialize SHAP value accumulator
        shap_values = np.zeros(self.n_supervoxels)
        shap_counts = np.zeros(self.n_supervoxels)

        # Permutation-based sampling
        for _ in range(n_samples):
            # Random permutation of supervoxels
            perm = np.random.permutation(self.n_supervoxels)

            # Compute marginal contributions
            prev_output = self._eval_masked(input_np, [])

            for i, sv_idx in enumerate(perm):
                # Include supervoxels up to current one
                included = perm[:i+1].tolist()
                curr_output = self._eval_masked(input_np, included)

                # Marginal contribution
                contribution = curr_output - prev_output
                shap_values[sv_idx] += contribution
                shap_counts[sv_idx] += 1

                prev_output = curr_output

        # Average SHAP values
        shap_values = shap_values / np.maximum(shap_counts, 1)

        # Reshape to supervoxel grid
        shap_values = shap_values.reshape(sv_shape)

        return shap_values

    def _eval_masked(
        self,
        input_np: np.ndarray,
        included_supervoxels: list
    ) -> float:
        """
        Evaluate model with only specified supervoxels included.

        Args:
            input_np: Input array (B, 1, 128, 128, 128)
            included_supervoxels: List of supervoxel indices to include

        Returns:
            Model output (scalar)
        """
        # Create masked input
        masked = np.zeros_like(input_np)

        # Fill in included supervoxels
        for sv_idx in included_supervoxels:
            # Convert flat index to 3D coordinates
            sv_coords = self._idx_to_coords(sv_idx)
            slices = self._coords_to_slices(sv_coords)
            masked[:, :, slices[0], slices[1], slices[2]] = \
                input_np[:, :, slices[0], slices[1], slices[2]]

        # Evaluate
        masked_tensor = torch.from_numpy(masked).float().to(self.device)
        with torch.no_grad():
            output = self.model(masked_tensor)
            if output.dim() > 1:
                return output.mean().item()
            return output.item()

    def _idx_to_coords(self, flat_idx: int) -> Tuple[int, int, int]:
        """Convert flat supervoxel index to 3D coordinates."""
        sz, sy, sx = self.sv_dims
        z = flat_idx // (sy * sx)
        y = (flat_idx % (sy * sx)) // sx
        x = flat_idx % sx
        return (z, y, x)

    def _coords_to_slices(
        self,
        coords: Tuple[int, int, int]
    ) -> Tuple[slice, slice, slice]:
        """Convert supervoxel coordinates to array slices."""
        sv = self.supervoxel_size
        z, y, x = coords
        return (
            slice(z * sv, (z + 1) * sv),
            slice(y * sv, (y + 1) * sv),
            slice(x * sv, (x + 1) * sv)
        )

    def _upsample_supervoxels(self, sv_shap: np.ndarray) -> np.ndarray:
        """
        Upsample supervoxel SHAP values to full resolution.

        Uses trilinear interpolation for smooth results.
        """
        # Current shape: (16, 16, 16) for supervoxel_size=8
        # Target shape: (128, 128, 128)
        zoom_factors = tuple(
            self.input_shape[i] / sv_shap.shape[i]
            for i in range(3)
        )

        full_shap = zoom(sv_shap, zoom_factors, order=1)

        # Add batch and channel dimensions
        full_shap = full_shap[np.newaxis, np.newaxis, ...]

        return full_shap


class IntegratedGradientsMRI:
    """
    Integrated Gradients for MRI explainability.

    Alternative to SHAP that uses gradient integration along path
    from baseline to input.
    """

    def __init__(
        self,
        model: nn.Module,
        baseline: Optional[torch.Tensor] = None,
        n_steps: int = 50,
        device: str = 'cuda'
    ):
        """
        Initialize Integrated Gradients.

        Args:
            model: LightMRI3D model
            baseline: Baseline input (default: zeros)
            n_steps: Number of integration steps
            device: Device to run on
        """
        self.device = device
        self.n_steps = n_steps

        # Extract MRI model if wrapped
        self.model = self._extract_mri_model(model)
        self.model.eval()

        # Default baseline: zeros
        if baseline is None:
            self.baseline = torch.zeros(1, 1, 128, 128, 128).to(device)
        else:
            self.baseline = baseline.to(device)

    def _extract_mri_model(self, model: nn.Module) -> nn.Module:
        """Extract LightMRI3D from wrapper if needed."""
        if hasattr(model, 'img_model'):
            return model.img_model.to(self.device)
        return model.to(self.device)

    def compute(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 0
    ) -> np.ndarray:
        """
        Compute Integrated Gradients.

        Args:
            input_tensor: Input MRI (B, 1, 128, 128, 128)
            target_class: Class to explain

        Returns:
            Attribution map (B, 1, 128, 128, 128)
        """
        input_tensor = input_tensor.to(self.device)

        # Expand baseline to match batch size
        baseline = self.baseline.expand_as(input_tensor)

        # Create interpolated inputs
        scaled_inputs = []
        for step in range(self.n_steps + 1):
            alpha = step / self.n_steps
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad_(True)
            scaled_inputs.append(interpolated)

        # Compute gradients at each step
        gradients = []
        for scaled_input in scaled_inputs:
            output = self.model(scaled_input)

            # Get score for target class
            if output.dim() == 2:
                score = output[:, target_class]
            else:
                score = output

            # Compute gradient
            self.model.zero_grad()
            grad = torch.autograd.grad(
                outputs=score.sum(),
                inputs=scaled_input,
                retain_graph=False,
                create_graph=False
            )[0]
            gradients.append(grad)

        # Average gradients (Riemann sum)
        avg_gradients = torch.stack(gradients).mean(dim=0)

        # Integrated gradients = (input - baseline) * avg_gradients
        attributions = (input_tensor - baseline) * avg_gradients

        # Take absolute value and normalize
        attributions = torch.abs(attributions)
        attr_flat = attributions.view(attributions.size(0), -1)
        attr_max = attr_flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1, 1)
        attributions = attributions / (attr_max + 1e-8)

        return attributions.detach().cpu().numpy()


def compute_shap_for_sample(
    model: nn.Module,
    input_path: str,
    background_paths: Optional[list] = None,
    supervoxel_size: int = 8,
    n_samples: int = 100,
    device: str = 'cuda'
) -> Tuple[np.ndarray, dict]:
    """
    Convenience function to compute SHAP values for a preprocessed MRI sample.

    Args:
        model: LightMRI3D model
        input_path: Path to preprocessed .npy file
        background_paths: Optional list of paths for background samples
        supervoxel_size: Size of supervoxels
        n_samples: Number of samples for SHAP
        device: Device to run on

    Returns:
        Tuple of (shap_array, metadata)
    """
    # Load input
    input_data = np.load(input_path)
    input_tensor = torch.from_numpy(input_data).float()
    if input_tensor.dim() == 4:
        input_tensor = input_tensor.unsqueeze(0)

    # Load background if provided
    background = None
    if background_paths:
        bg_list = []
        for path in background_paths[:10]:  # Limit to 10
            bg_data = np.load(path)
            bg_list.append(bg_data)
        background = torch.from_numpy(np.stack(bg_list)).float()

    # Initialize explainer
    explainer = SHAPExplainerMRI(
        model,
        background_data=background,
        supervoxel_size=supervoxel_size,
        device=device
    )

    # Compute SHAP values
    result = explainer.compute(input_tensor, n_samples=n_samples)

    metadata = {
        'method': result.method,
        'supervoxel_size': result.supervoxel_size,
        'base_value': result.base_value,
        'prediction': result.prediction,
        'shap_shape': result.shap_values.shape,
        'shap_min': float(result.shap_values.min()),
        'shap_max': float(result.shap_values.max()),
    }

    return result.shap_values, metadata
