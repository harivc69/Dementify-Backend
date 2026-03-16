"""
MRI Explainability Module

Provides GradCAM and SHAP-based explanations for MRI classification models,
with proper reverse transforms to overlay heatmaps on original brain images.
"""

from .gradcam_mri import GradCAMMRI
from .reverse_transform import ReverseTransformer
from .visualization import ExplainabilityVisualizer

__all__ = [
    'GradCAMMRI',
    'ReverseTransformer',
    'ExplainabilityVisualizer',
]

# Optional imports (may not be present in all deployments)
try:
    from .validation import ExplainabilityValidator, ValidationError
    __all__.extend(['ExplainabilityValidator', 'ValidationError'])
except ImportError:
    pass

try:
    from .nifti_mapper import NIfTIMapper
    __all__.append('NIfTIMapper')
except ImportError:
    pass

try:
    from .shap_mri import SHAPExplainerMRI
    __all__.append('SHAPExplainerMRI')
except ImportError:
    pass
