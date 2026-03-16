#!/usr/bin/env python3
"""
MRI + Tabular Cognitive Impairment Classifier - Frontend API v2.0

Clean API for the overseas frontend team to integrate the hierarchical
dementia classifier with MRI and 187 tabular features.

Features:
    - Accepts NIfTI (.nii / .nii.gz) MRI input
    - Returns JSON with hierarchical predictions
    - GradCAM++ heatmap output as NIfTI for overlay
    - Skull-stripped brain NIfTI for frontend overlay base
    - SHAP explanations for 187 tabular features + MRI importance
    - Batch processing support

Quick Start:
    from api import MRICognitiveClassifierAPI

    api = MRICognitiveClassifierAPI()
    result = api.predict('scan.nii.gz', {'his_NACCAGE': 72, ...}, output_dir='output/')
    result_with_shap = api.predict_with_explanations(
        'scan.nii.gz', {'his_NACCAGE': 72, ...}, output_dir='output/'
    )

Author: UIUC Research Team
"""

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

warnings.filterwarnings('ignore')

# Resolve package directory
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add nmed2024 to path
_nmed_dev = os.path.join(PACKAGE_DIR, 'nmed2024', 'dev')
_nmed_root = os.path.join(PACKAGE_DIR, 'nmed2024')
if os.path.isdir(_nmed_dev) and _nmed_dev not in sys.path:
    sys.path.insert(0, _nmed_dev)
if os.path.isdir(_nmed_root) and _nmed_root not in sys.path:
    sys.path.insert(0, _nmed_root)

from inference_pipeline import HierarchicalInferencePipeline


class MRICognitiveClassifierAPI:
    """
    Production API for the MRI + Tabular Hierarchical Cognitive Classifier.

    Provides a clean interface for frontend integration with:
    - NIfTI MRI input and NIfTI heatmap output
    - Fast inference across 3 hierarchical stages
    - SHAP explanations for tabular features + MRI importance
    - Batch processing capabilities

    Example:
        >>> api = MRICognitiveClassifierAPI()
        >>>
        >>> # Fast prediction (no SHAP, optional heatmap)
        >>> result = api.predict('scan.nii.gz', features, output_dir='output/')
        >>> print(result['final_diagnosis'])
        >>>
        >>> # Full prediction with explanations
        >>> result = api.predict_with_explanations(
        ...     'scan.nii.gz', features, output_dir='output/'
        ... )
        >>> print(result['feature_importance']['top_features'])
    """

    # Dementia subtype full names
    SUBTYPE_NAMES = {
        'AD': "Alzheimer's Disease",
        'LBD': 'Lewy Body Dementia',
        'VD': 'Vascular Dementia',
        'FTD': 'Frontotemporal Dementia',
        'PRD': 'Prion Disease (CJD)',
        'NPH': 'Normal Pressure Hydrocephalus',
        'SEF': 'Systemic/Environmental Factors',
        'PSY': 'Psychiatric Causes',
        'ODE': 'Other Dementia Etiologies'
    }

    # Stage 2 class names
    COGNITIVE_STATUS_NAMES = {
        'NC': 'Normal Cognition',
        'MCI': 'Mild Cognitive Impairment',
        'IMCI': 'MCI with Functional Impairment'
    }

    def __init__(
        self,
        models_dir: Optional[str] = None,
        configs_dir: Optional[str] = None,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the MRI classifier API.

        Auto-discovers model checkpoints and config files relative to the
        package directory. Override with explicit paths if needed.

        Args:
            models_dir: Directory containing stage1/2/3 .pt files.
                        Default: <package_dir>/models/
            configs_dir: Directory containing stage1/2/3 .toml configs.
                         Default: <package_dir>/configs/
            device: 'cuda' or 'cpu'. Auto-detected if None.
            verbose: Whether to print loading messages.
        """
        models_dir = models_dir or os.path.join(PACKAGE_DIR, 'models')
        configs_dir = configs_dir or os.path.join(PACKAGE_DIR, 'configs')

        stage1_ckpt = os.path.join(models_dir, 'stage1_de_vs_non_de.pt')
        stage2_ckpt = os.path.join(models_dir, 'stage2_nc_mci_imci.pt')
        stage3_ckpt = os.path.join(models_dir, 'stage3_dementia_subtypes.pt')

        stage1_config = os.path.join(configs_dir, 'stage1_de_mri_config.toml')
        stage2_config = os.path.join(configs_dir, 'stage2_3way_mri_config.toml')
        stage3_config = os.path.join(configs_dir, 'stage3_10class_mri_config.toml')

        if not verbose:
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

        self.pipeline = HierarchicalInferencePipeline(
            stage1_checkpoint=stage1_ckpt,
            stage2_checkpoint=stage2_ckpt,
            stage3_checkpoint=stage3_ckpt,
            stage1_config=stage1_config,
            stage2_config=stage2_config,
            stage3_config=stage3_config,
            device=device or 'cuda'
        )

        if not verbose:
            sys.stdout.close()
            sys.stdout = old_stdout

        self.device = self.pipeline.device

    def predict(
        self,
        mri_path: str,
        features: Union[Dict[str, Any], str],
        output_dir: Optional[str] = None,
        generate_heatmap: bool = True
    ) -> Dict[str, Any]:
        """
        Fast hierarchical inference (3 stages, optional heatmap, no SHAP).

        Args:
            mri_path: Path to MRI scan (.nii or .nii.gz)
            features: Dict of 187 tabular features, or path to JSON file.
                      Missing features: set to null/-4 (model handles via attention masking).
            output_dir: Directory to save NIfTI heatmap outputs.
                        Required if generate_heatmap=True.
            generate_heatmap: Whether to generate GradCAM++ heatmap (default: True).

        Returns:
            Dict with prediction results:
            {
                'timestamp': str,
                'mri_path': str,
                'stage1': {'prediction': str, 'de_probability': float, 'confidence': float},
                'stage2': {...} or None,
                'stage3': {...} or None,
                'final_diagnosis': str,
                'heatmaps': {
                    'heatmap_nifti': str,    # Path to heatmap.nii.gz
                    'brain_nifti': str,      # Path to brain.nii.gz
                    'overlay_nifti': str,    # Path to overlay.nii.gz
                    'method': 'gradcam_pp'
                }
            }
        """
        features = self._load_features(features)

        if generate_heatmap and output_dir is None:
            output_dir = os.path.join(os.path.dirname(mri_path), 'output')

        return self.pipeline.predict(
            mri_path=mri_path,
            features=features,
            generate_heatmap=generate_heatmap,
            output_dir=output_dir
        )

    def predict_with_explanations(
        self,
        mri_path: str,
        features: Union[Dict[str, Any], str],
        output_dir: str,
        n_top_features: int = 20,
        nsamples: int = 100
    ) -> Dict[str, Any]:
        """
        Full inference with tabular SHAP explanations + MRI heatmap.

        This is slower than predict() due to SHAP computation (~3-5 minutes).

        Args:
            mri_path: Path to MRI scan (.nii or .nii.gz)
            features: Dict of 187 tabular features, or path to JSON file.
            output_dir: Directory to save NIfTI heatmap outputs (required).
            n_top_features: Number of top SHAP features to include (default: 20).
            nsamples: SHAP evaluation samples (default: 100, increase for precision).

        Returns:
            Dict with predictions + explanations:
            {
                ... (all fields from predict()),
                'feature_importance': {
                    'tabular_shap': {
                        'feature_names': [str, ...],
                        'shap_values': [float, ...],
                        'base_value': float
                    },
                    'mri_importance': float,
                    'top_features': [
                        {
                            'feature': str,
                            'description': str,
                            'shap_value': float,
                            'direction': 'increases' | 'decreases',
                            'importance': float
                        }, ...
                    ]
                }
            }
        """
        features = self._load_features(features)

        return self.pipeline.predict_with_explanations(
            mri_path=mri_path,
            features=features,
            output_dir=output_dir,
            n_top_features=n_top_features,
            nsamples=nsamples
        )

    def predict_batch(
        self,
        patients: List[Dict[str, Any]],
        output_dir: str,
        generate_heatmap: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Batch prediction for multiple patients.

        Args:
            patients: List of dicts, each with 'mri_path' and 'features' keys.
                      Example: [
                          {'mri_path': 'scan1.nii.gz', 'features': {...}},
                          {'mri_path': 'scan2.nii.gz', 'features': {...}},
                      ]
            output_dir: Base directory for outputs. Each patient gets a subdirectory.
            generate_heatmap: Whether to generate heatmaps (default: True).

        Returns:
            List of prediction result dicts.
        """
        results = []

        for i, patient in enumerate(patients):
            mri_path = patient['mri_path']
            features = self._load_features(patient['features'])
            patient_output = os.path.join(output_dir, f'patient_{i}')

            try:
                result = self.pipeline.predict(
                    mri_path=mri_path,
                    features=features,
                    generate_heatmap=generate_heatmap,
                    output_dir=patient_output
                )
                result['patient_index'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'patient_index': i,
                    'mri_path': mri_path,
                    'error': str(e)
                })

        return results

    def get_feature_info(self) -> Dict[str, Any]:
        """
        Return information about the 187 tabular features + MRI.

        Returns:
            Dict with:
            {
                'total_features': 188,
                'tabular_features': 187,
                'mri_features': 1,
                'features': [
                    {'name': str, 'category': str, 'description': str}, ...
                ]
            }
        """
        # Get tabular feature names from stage1 model
        feature_cols = list(self.pipeline.stage1_model.src_modalities.keys())
        tabular_cols = [c for c in feature_cols if not c.startswith('img_')]

        features = []
        for name in tabular_cols:
            desc = self.pipeline._get_feature_description(name)
            # Extract category from description
            if ':' in desc:
                category, short_name = desc.split(':', 1)
            else:
                category = 'Other'
                short_name = name
            features.append({
                'name': name,
                'category': category.strip(),
                'description': desc
            })

        # Add MRI feature
        features.append({
            'name': 'img_MRI_T1_1',
            'category': 'Brain MRI',
            'description': 'Brain MRI: T1-weighted structural MRI scan (NIfTI format)'
        })

        return {
            'total_features': len(features),
            'tabular_features': len(tabular_cols),
            'mri_features': 1,
            'features': features
        }

    def _load_features(self, features: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """Load features from dict or JSON file path."""
        if isinstance(features, str):
            with open(features, 'r') as f:
                return json.load(f)
        return features

    def save_result(self, result: Dict, output_path: str):
        """Save prediction result to JSON file."""
        self.pipeline.save_result(result, output_path)
