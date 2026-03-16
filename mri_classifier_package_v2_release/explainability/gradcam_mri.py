"""
GradCAM Implementation for LightMRI3D Model.

Generates class activation maps for 3D MRI classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple, List, Any
from dataclasses import dataclass
import warnings


@dataclass
class GradCAMResult:
    """Result from GradCAM computation."""
    cam: torch.Tensor  # Raw CAM (B, 1, D, H, W)
    activations: torch.Tensor  # Layer activations
    gradients: torch.Tensor  # Layer gradients
    target_class: int
    target_layer: str


class GradCAMMRI:
    """
    GradCAM for LightMRI3D model.

    Target layers for LightMRI3D:
    - 'bn4': After stage 4 pointwise conv -> (B, 256, 8, 8, 8) - RECOMMENDED
    - 'bn3': After stage 3 pointwise conv -> (B, 128, 16, 16, 16)
    - 'bn2': After stage 2 pointwise conv -> (B, 64, 32, 32, 32)

    The heatmap from bn4 has 8x8x8 spatial resolution, which is then
    upsampled to match the input/original resolution.
    """

    # Target layer options for LightMRI3D
    TARGET_LAYERS = {
        'stage4': 'bn4',   # (B, 256, 8, 8, 8) - best for localization
        'stage3': 'bn3',   # (B, 128, 16, 16, 16)
        'stage2': 'bn2',   # (B, 64, 32, 32, 32)
    }

    def __init__(
        self,
        model: nn.Module,
        target_layer: str = 'bn4',
        device: str = 'cuda',
        use_gradcam_pp: bool = False
    ):
        """
        Initialize GradCAM for LightMRI3D.

        Args:
            model: LightMRI3D model (or wrapper containing img_model)
            target_layer: Layer name to compute GradCAM on ('bn4', 'bn3', 'bn2')
            device: Device to run on ('cuda' or 'cpu')
            use_gradcam_pp: Use GradCAM++ weighting (better for multiple objects)
        """
        self.device = device
        self.target_layer = target_layer
        self.use_gradcam_pp = use_gradcam_pp

        # Extract LightMRI3D model if wrapped
        self.model = self._extract_mri_model(model)
        self.model.eval()

        # Storage for hooks
        self._activations: Dict[str, torch.Tensor] = {}
        self._gradients: Dict[str, torch.Tensor] = {}
        self._handles: List[Any] = []

        # Register hooks
        self._register_hooks()

    def _extract_mri_model(self, model: nn.Module) -> nn.Module:
        """Extract LightMRI3D from wrapper if needed."""
        # Check if model has img_model attribute (ImagingModelWrapper)
        if hasattr(model, 'img_model'):
            return model.img_model.to(self.device)

        # Check if model has modules_emb_src (Transformer)
        if hasattr(model, 'modules_emb_src'):
            for key, module in model.modules_emb_src.items():
                if hasattr(module, 'img_model'):
                    return module.img_model.to(self.device)

        # Assume direct LightMRI3D model
        return model.to(self.device)

    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        target_module = self._get_target_module()

        if target_module is None:
            raise ValueError(f"Target layer '{self.target_layer}' not found in model")

        # Forward hook to capture activations
        def forward_hook(module, input, output):
            self._activations[self.target_layer] = output.detach()

        # Backward hook to capture gradients
        def backward_hook(module, grad_input, grad_output):
            self._gradients[self.target_layer] = grad_output[0].detach()

        # Register hooks
        handle_fwd = target_module.register_forward_hook(forward_hook)
        handle_bwd = target_module.register_full_backward_hook(backward_hook)

        self._handles.extend([handle_fwd, handle_bwd])

    def _get_target_module(self) -> Optional[nn.Module]:
        """Get the target layer module from LightMRI3D."""
        # Try direct attribute access
        if hasattr(self.model, self.target_layer):
            return getattr(self.model, self.target_layer)

        # Try searching through named modules
        for name, module in self.model.named_modules():
            if name == self.target_layer or name.endswith(f'.{self.target_layer}'):
                return module

        return None

    def compute(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        retain_graph: bool = False
    ) -> GradCAMResult:
        """
        Compute GradCAM for input MRI.

        Args:
            input_tensor: Input MRI tensor (B, 1, 128, 128, 128)
            target_class: Class index to compute CAM for (default: predicted class)
            retain_graph: Whether to retain computation graph

        Returns:
            GradCAMResult with CAM and metadata
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)

        # Clear previous activations/gradients
        self._activations.clear()
        self._gradients.clear()

        # Forward pass with gradients enabled (needed for GradCAM even in eval mode)
        with torch.enable_grad():
            output = self.model(input_tensor)

            # Handle different output formats
            if target_class is None:
                # Use index of max output as target
                if output.dim() == 2:
                    target_class = output.argmax(dim=1).item()
                else:
                    target_class = 0

            # Get score for target class
            if output.dim() == 2:
                score = output[:, target_class]
            else:
                score = output

            # Backward pass
            self.model.zero_grad()
            score.sum().backward(retain_graph=retain_graph)

        # Get activations and gradients
        if self.target_layer not in self._activations:
            raise RuntimeError(f"No activations captured for {self.target_layer}")
        if self.target_layer not in self._gradients:
            raise RuntimeError(f"No gradients captured for {self.target_layer}")

        activations = self._activations[self.target_layer]
        gradients = self._gradients[self.target_layer]

        # Compute CAM
        if self.use_gradcam_pp:
            cam = self._compute_gradcam_pp(activations, gradients)
        else:
            cam = self._compute_gradcam(activations, gradients)

        return GradCAMResult(
            cam=cam,
            activations=activations,
            gradients=gradients,
            target_class=target_class,
            target_layer=self.target_layer
        )

    def _compute_gradcam(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute standard GradCAM.

        CAM = ReLU(sum_c(weight_c * activation_c))
        where weight_c = global_avg_pool(gradient_c)
        """
        # Global average pooling of gradients to get weights
        # (B, C, D, H, W) -> (B, C, 1, 1, 1)
        weights = gradients.mean(dim=(2, 3, 4), keepdim=True)

        # Weighted combination of activations
        # (B, C, D, H, W) * (B, C, 1, 1, 1) -> (B, C, D, H, W) -> (B, 1, D, H, W)
        cam = (weights * activations).sum(dim=1, keepdim=True)

        # ReLU to keep only positive contributions
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = self._normalize_cam(cam)

        return cam

    def _compute_gradcam_pp(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute GradCAM++ with improved weighting.

        Better at handling multiple discriminative regions.
        """
        # Compute alpha weights (GradCAM++ specific)
        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)

        # Compute alpha numerator and denominator
        sum_activations = activations.sum(dim=(2, 3, 4), keepdim=True)
        alpha_numer = grad_2
        alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8

        alpha = alpha_numer / alpha_denom
        alpha = alpha * F.relu(gradients)  # Only positive gradients

        # Compute weights
        weights = alpha.sum(dim=(2, 3, 4), keepdim=True)

        # Weighted combination
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = self._normalize_cam(cam)

        return cam

    def _normalize_cam(self, cam: torch.Tensor) -> torch.Tensor:
        """Normalize CAM to [0, 1] per sample."""
        b = cam.shape[0]
        cam_flat = cam.view(b, -1)
        cam_min = cam_flat.min(dim=1, keepdim=True)[0]
        cam_max = cam_flat.max(dim=1, keepdim=True)[0]

        # Avoid division by zero
        cam_range = cam_max - cam_min
        cam_range[cam_range < 1e-8] = 1e-8

        cam_norm = (cam_flat - cam_min) / cam_range
        return cam_norm.view_as(cam)

    def __del__(self):
        """Remove hooks when object is deleted."""
        for handle in self._handles:
            handle.remove()


class ModelWrapper(nn.Module):
    """
    Wrapper for ADRDModel to enable direct GradCAM computation.

    Extracts the image processing path and wraps it for GradCAM.
    """

    def __init__(
        self,
        adrd_model: nn.Module,
        target_label: str = 'DE',
        device: str = 'cuda'
    ):
        super().__init__()
        self.device = device
        self.target_label = target_label

        # Extract components from ADRDModel
        self.net = adrd_model.net_.to(device)

        # Find the image embedding module
        self.img_module = None
        if hasattr(self.net, 'modules_emb_src'):
            for key, module in self.net.modules_emb_src.items():
                if 'img' in key.lower() or 'mri' in key.lower():
                    self.img_module = module
                    break

        if self.img_module is None:
            raise ValueError("Could not find image module in ADRDModel")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through image module only.

        Args:
            x: Input MRI tensor (B, 1, 128, 128, 128)

        Returns:
            Feature embedding from image model
        """
        return self.img_module(x)

    def get_mri_model(self) -> nn.Module:
        """Get the underlying MRI model (LightMRI3D)."""
        if hasattr(self.img_module, 'img_model'):
            return self.img_module.img_model
        return self.img_module


def compute_gradcam_for_sample(
    model: nn.Module,
    input_path: str,
    target_class: int = 0,
    target_layer: str = 'bn4',
    device: str = 'cuda'
) -> Tuple[np.ndarray, dict]:
    """
    Convenience function to compute GradCAM for a preprocessed MRI sample.

    Args:
        model: LightMRI3D model or ADRDModel
        input_path: Path to preprocessed .npy file
        target_class: Class index to explain
        target_layer: Layer to compute GradCAM on
        device: Device to run on

    Returns:
        Tuple of (cam_array, metadata)
    """
    # Load preprocessed input
    input_data = np.load(input_path)
    input_tensor = torch.from_numpy(input_data).float()

    # Add batch dimension if needed
    if input_tensor.dim() == 4:
        input_tensor = input_tensor.unsqueeze(0)

    # Initialize GradCAM
    gradcam = GradCAMMRI(model, target_layer=target_layer, device=device)

    # Compute
    result = gradcam.compute(input_tensor, target_class=target_class)

    # Convert to numpy
    cam_array = result.cam.cpu().numpy()

    metadata = {
        'target_class': result.target_class,
        'target_layer': result.target_layer,
        'input_shape': input_tensor.shape,
        'cam_shape': cam_array.shape,
        'cam_min': float(cam_array.min()),
        'cam_max': float(cam_array.max()),
    }

    return cam_array, metadata
