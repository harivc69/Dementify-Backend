#!/usr/bin/env python3
"""
MRI + Tabular Hierarchical Inference Pipeline v2.0

Production inference script for the overseas team frontend integration.

Input:
    - MRI scan: .nii or .nii.gz format
    - Tabular features: JSON object with 187 NACC features

Output:
    - JSON with hierarchical predictions, heatmap paths, and SHAP explanations

Usage:
    python inference_pipeline.py \
        --mri_path /path/to/scan.nii.gz \
        --features_json /path/to/features.json \
        --output_dir /path/to/output/ \
        --generate_heatmap

API Example:
    from inference_pipeline import HierarchicalInferencePipeline

    pipeline = HierarchicalInferencePipeline(
        stage1_checkpoint='models/stage1_de_vs_non_de.pt',
        stage2_checkpoint='models/stage2_nc_mci_imci.pt',
        stage3_checkpoint='models/stage3_dementia_subtypes.pt'
    )

    result = pipeline.predict(
        mri_path='scan.nii.gz',
        features={'his_NACCAGE': 72, 'his_SEX': 1, ...},
        generate_heatmap=True,
        output_dir='output/'
    )
"""

import os
import sys
import json
import argparse
import warnings
import torch
import numpy as np
import nibabel as nib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from scipy.ndimage import zoom, gaussian_filter

# Resolve package-relative paths for imports
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# Search for nmed2024 in package dir first, then parent (development layout)
_search_dirs = [PACKAGE_DIR, os.path.dirname(PACKAGE_DIR)]
for _search in _search_dirs:
    _nmed_dev = os.path.join(_search, 'nmed2024', 'dev')
    _nmed_root = os.path.join(_search, 'nmed2024')
    if os.path.isdir(_nmed_dev):
        if _nmed_dev not in sys.path:
            sys.path.insert(0, _nmed_dev)
        if _nmed_root not in sys.path:
            sys.path.insert(0, _nmed_root)
        break

from adrd.model import ADRDModel
from data.dataset_csv import CSVDataset

# SHAP for explainability (optional)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class MRIPreprocessor:
    """Preprocess .nii/.nii.gz to model-ready format."""

    def __init__(self, target_shape: Tuple[int, int, int] = (128, 128, 128)):
        self.target_shape = target_shape

    def load_nifti(self, nifti_path: str) -> Tuple[np.ndarray, np.ndarray, Tuple]:
        """Load NIfTI file and return data, affine, original shape."""
        img = nib.load(nifti_path)
        data = img.get_fdata()
        affine = img.affine
        original_shape = data.shape
        return data, affine, original_shape

    def preprocess(self, nifti_path: str) -> Tuple[np.ndarray, dict]:
        """
        Preprocess NIfTI to model input format with skull stripping.

        Returns:
            - preprocessed array: (1, 128, 128, 128) normalized to [0, 1]
            - metadata: dict with original shape, affine, brain_mask for heatmap mapping
        """
        data, affine, original_shape = self.load_nifti(nifti_path)

        # Handle 4D (some scans have time dimension)
        if len(data.shape) == 4:
            data = data[:, :, :, 0]

        # Skull strip
        try:
            sys.path.insert(0, PACKAGE_DIR)
            from skull_strip import skull_strip_array_safe
            data, brain_mask_orig, skull_stripped = skull_strip_array_safe(
                data, affine, no_gpu=False
            )
        except Exception as e:
            warnings.warn(f"Skull stripping unavailable: {e}")
            brain_mask_orig = data > 0
            skull_stripped = False

        # Store skull-stripped data in original space for brain.nii.gz output
        skull_stripped_original = data.copy()

        # Compute zoom factors
        zoom_factors = tuple(t / o for t, o in zip(self.target_shape, data.shape[:3]))

        # Resize to target shape
        resized = zoom(data, zoom_factors, order=1)  # Bilinear interpolation
        brain_mask_resized = zoom(brain_mask_orig.astype(np.float32), zoom_factors, order=0) > 0.5

        # Normalize to [0, 1]
        data_min, data_max = resized.min(), resized.max()
        if data_max > data_min:
            normalized = (resized - data_min) / (data_max - data_min)
        else:
            normalized = np.zeros_like(resized)

        # Add channel dimension: (128, 128, 128) -> (1, 128, 128, 128)
        preprocessed = normalized[np.newaxis, ...].astype(np.float32)

        metadata = {
            'original_shape': original_shape,
            'affine': affine.tolist(),
            'zoom_factors': zoom_factors,
            'data_min': float(data_min),
            'data_max': float(data_max),
            'nifti_path': nifti_path,
            'brain_mask': brain_mask_resized,
            'skull_stripped': skull_stripped,
            'skull_stripped_original': skull_stripped_original,
            'brain_mask_original': brain_mask_orig,
        }

        return preprocessed, metadata


class HierarchicalInferencePipeline:
    """
    Hierarchical 3-stage inference pipeline for MRI + tabular features.

    Stage 1: DE vs Non-DE (Dementia vs Normal/MCI/iMCI)
    Stage 2: If Non-DE -> NC vs MCI vs iMCI classification
    Stage 3: If DE -> Dementia subtype classification (AD, LBD, VD, FTD, etc.)
    """

    STAGE1_LABELS = ['Non-DE', 'DE']
    STAGE2_LABELS = ['NC', 'IMCI', 'MCI']
    STAGE3_LABELS = ['AD', 'LBD', 'VD', 'PRD', 'FTD', 'NPH', 'SEF', 'PSY', 'ODE']

    # Feature prefix to category mapping for descriptions
    FEATURE_PREFIXES = {
        'his_': 'Medical History',
        'med_': 'Medication',
        'ph_': 'Physical Measurement',
        'bat_': 'Cognitive Test',
        'exam_': 'Neurological Exam',
        'cvd_': 'Cardiovascular',
        'updrs_': 'UPDRS Score',
        'npiq_': 'Neuropsychiatric',
        'gds_': 'Depression Scale',
        'faq_': 'Functional Assessment',
        'img_': 'Brain MRI',
    }

    def __init__(
        self,
        stage1_checkpoint: str,
        stage2_checkpoint: str,
        stage3_checkpoint: str,
        stage1_config: str = None,
        stage2_config: str = None,
        stage3_config: str = None,
        device: str = 'cuda'
    ):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.preprocessor = MRIPreprocessor()

        # Default config paths (relative to package directory)
        base_dir = PACKAGE_DIR
        self.stage1_config = stage1_config or os.path.join(base_dir, 'configs', 'stage1_de_mri_config.toml')
        self.stage2_config = stage2_config or os.path.join(base_dir, 'configs', 'stage2_3way_mri_config.toml')
        self.stage3_config = stage3_config or os.path.join(base_dir, 'configs', 'stage3_10class_mri_config.toml')

        # Fallback to configs_mri if configs/ doesn't exist (development layout)
        if not os.path.exists(self.stage1_config):
            configs_mri = os.path.join(base_dir, 'configs_mri')
            if os.path.isdir(configs_mri):
                self.stage1_config = stage1_config or os.path.join(configs_mri, 'stage1_de_mri_config.toml')
                self.stage2_config = stage2_config or os.path.join(configs_mri, 'stage2_3way_mri_config.toml')
                self.stage3_config = stage3_config or os.path.join(configs_mri, 'stage3_10class_mri_config.toml')

        # Load models
        print(f"Loading models on {self.device}...")
        self.stage1_model = self._load_model(stage1_checkpoint, self.stage1_config)
        self.stage2_model = self._load_model(stage2_checkpoint, self.stage2_config)
        self.stage3_model = self._load_model(stage3_checkpoint, self.stage3_config)
        print("Models loaded successfully.")

    def _load_model(self, checkpoint_path: str, config_path: str) -> ADRDModel:
        """Load a trained model from checkpoint."""
        import tomli

        with open(config_path, 'rb') as f:
            config = tomli.load(f)

        # Build feature modalities from config
        feature_modalities = {}
        for key, val in config.get('feature', {}).items():
            feature_modalities[key] = val

        label_modalities = {}
        for key, val in config.get('label', {}).items():
            label_modalities[key] = val

        # Initialize model
        model = ADRDModel(
            src_modalities=feature_modalities,
            tgt_modalities=label_modalities,
            label_fractions={k: 0.5 for k in label_modalities},
            d_model=128,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            device=self.device,
            img_net='LightMRI',
            imgnet_layers=2,
            img_size=(128, 128, 128),
            train_imgnet=False,
            batch_size=1,
            _dataloader_num_workers=0
        )

        # Load checkpoint
        model.load(checkpoint_path)
        model.net_.eval()

        return model

    def _prepare_features(
        self,
        mri_array: np.ndarray,
        tabular_features: Dict[str, Any],
        feature_modalities: Dict
    ) -> Dict[str, Any]:
        """Prepare feature dict for model input."""
        features = {}

        for key in feature_modalities:
            if key.startswith('img_'):
                features[key] = mri_array
            elif key in tabular_features:
                features[key] = tabular_features[key]
            else:
                features[key] = None

        return features

    def _get_feature_description(self, name: str) -> str:
        """Get human-readable description for a feature."""
        for prefix, category in self.FEATURE_PREFIXES.items():
            if name.startswith(prefix):
                return f"{category}: {name[len(prefix):]}"
        return name

    def predict(
        self,
        mri_path: str,
        features: Dict[str, Any],
        generate_heatmap: bool = False,
        output_dir: Optional[str] = None,
        heatmap_method: str = 'gradcam'
    ) -> Dict[str, Any]:
        """
        Run hierarchical prediction.

        Args:
            mri_path: Path to .nii or .nii.gz MRI scan
            features: Dict of 187 tabular features
            generate_heatmap: Whether to generate GradCAM++ heatmap
            output_dir: Directory for heatmap outputs
            heatmap_method: 'gradcam' (default, uses GradCAM++)

        Returns:
            Dict with prediction results and heatmap paths
        """
        timestamp = datetime.now().isoformat()

        # Preprocess MRI
        mri_array, mri_metadata = self.preprocessor.preprocess(mri_path)

        # Stage 1: DE vs Non-DE
        stage1_features = self._prepare_features(
            mri_array, features, self.stage1_model.src_modalities
        )
        # predict_proba returns (logits_list, proba_list) tuple
        _, stage1_proba = self.stage1_model.predict_proba([stage1_features], _batch_size=1)

        # Get DE probability (sigmoid output for binary DE label)
        de_prob = float(stage1_proba[0].get('DE', 0.5))
        is_dementia = de_prob >= 0.5

        result = {
            'timestamp': timestamp,
            'mri_path': mri_path,
            'stage1': {
                'prediction': 'DE' if is_dementia else 'Non-DE',
                'de_probability': de_prob,
                'confidence': de_prob if is_dementia else (1 - de_prob)
            },
            'stage2': None,
            'stage3': None,
            'final_diagnosis': None,
            'heatmaps': {}
        }

        if is_dementia:
            # Stage 3: Dementia subtype classification
            stage3_features = self._prepare_features(
                mri_array, features, self.stage3_model.src_modalities
            )
            # predict_proba returns (logits_list, proba_list) tuple
            _, stage3_proba = self.stage3_model.predict_proba([stage3_features], _batch_size=1)

            subtype_probs = {}
            for label in self.STAGE3_LABELS:
                if label in stage3_proba[0]:
                    subtype_probs[label] = float(stage3_proba[0][label])

            if subtype_probs:
                top_subtype = max(subtype_probs, key=subtype_probs.get)
                result['stage3'] = {
                    'prediction': top_subtype,
                    'probabilities': subtype_probs,
                    'confidence': subtype_probs[top_subtype]
                }
                result['final_diagnosis'] = top_subtype
        else:
            # Stage 2: NC vs MCI vs iMCI
            stage2_features = self._prepare_features(
                mri_array, features, self.stage2_model.src_modalities
            )
            # predict_proba returns (logits_list, proba_list) tuple
            _, stage2_proba = self.stage2_model.predict_proba([stage2_features], _batch_size=1)

            class_probs = {}
            for label in self.STAGE2_LABELS:
                if label in stage2_proba[0]:
                    class_probs[label] = float(stage2_proba[0][label])

            if class_probs:
                top_class = max(class_probs, key=class_probs.get)
                result['stage2'] = {
                    'prediction': top_class,
                    'probabilities': class_probs,
                    'confidence': class_probs[top_class]
                }
                result['final_diagnosis'] = top_class

        # Generate heatmap if requested
        if generate_heatmap and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            heatmap_paths = self._generate_heatmap(
                mri_array,
                mri_metadata,
                mri_path,
                output_dir,
                is_dementia
            )
            result['heatmaps'] = heatmap_paths

        return result

    def _generate_heatmap(
        self,
        mri_array: np.ndarray,
        mri_metadata: dict,
        original_nifti_path: str,
        output_dir: str,
        is_dementia: bool
    ) -> Dict[str, str]:
        """
        Generate GradCAM++ heatmap and save as NIfTI.

        Uses bn3 target layer (16x16x16) for higher resolution attribution,
        GradCAM++ for multi-region sensitivity, and Gaussian smoothing.

        Returns dict with paths to heatmap.nii.gz, brain.nii.gz, overlay.nii.gz
        """
        sys.path.insert(0, PACKAGE_DIR)
        from explainability.gradcam_mri import GradCAMMRI

        # Use the relevant model based on stage
        model = self.stage3_model if is_dementia else self.stage1_model
        target_class = 0

        # Initialize GradCAM++ with bn3 (16x16x16) for higher resolution
        gradcam = GradCAMMRI(
            model=model.net_,
            target_layer='bn3',
            device=self.device,
            use_gradcam_pp=True
        )

        # Compute attribution
        input_tensor = torch.from_numpy(mri_array).unsqueeze(0).to(self.device)
        result = gradcam.compute(input_tensor, target_class=target_class)

        # Extract numpy CAM from GradCAMResult dataclass
        cam = result.cam.cpu().numpy()
        while cam.ndim > 3:
            cam = cam[0]

        # Resize attribution to original NIfTI shape with cubic interpolation + smoothing
        original_shape = tuple(mri_metadata['original_shape'][:3])
        zoom_factors = tuple(o / t for o, t in zip(original_shape, cam.shape))
        cam_original = zoom(cam, zoom_factors, order=3)  # Cubic spline

        # Gaussian smoothing for cleaner heatmaps
        cam_original = gaussian_filter(cam_original, sigma=2.0)

        # Normalize to [0, 1]
        c_min, c_max = cam_original.min(), cam_original.max()
        if c_max - c_min > 1e-8:
            cam_original = (cam_original - c_min) / (c_max - c_min)

        # Apply brain mask to heatmap (restrict attribution to brain tissue)
        brain_mask = mri_metadata.get('brain_mask')
        brain_mask_orig = None
        if brain_mask is not None:
            mask_zoom = tuple(o / t for o, t in zip(original_shape, brain_mask.shape))
            brain_mask_orig = zoom(brain_mask.astype(np.float32), mask_zoom, order=0) > 0.5
            cam_original[~brain_mask_orig] = 0
            c_max = cam_original.max()
            if c_max > 1e-8:
                cam_original = cam_original / c_max

        # Store cam_masked for MRI importance computation
        self._last_cam_masked = cam_original.copy()
        self._last_brain_mask_orig = brain_mask_orig

        # Load original NIfTI for affine
        original_img = nib.load(original_nifti_path)
        affine = original_img.affine

        # Save heatmap as NIfTI
        heatmap_path = os.path.join(output_dir, 'heatmap.nii.gz')
        heatmap_nii = nib.Nifti1Image(cam_original.astype(np.float32), affine)
        nib.save(heatmap_nii, heatmap_path)

        # Save skull-stripped brain as NIfTI for frontend overlay base
        brain_path = os.path.join(output_dir, 'brain.nii.gz')
        skull_stripped_original = mri_metadata.get('skull_stripped_original')
        if skull_stripped_original is not None:
            brain_nii = nib.Nifti1Image(skull_stripped_original.astype(np.float32), affine)
            nib.save(brain_nii, brain_path)
        else:
            # Fallback: use original data
            original_data = original_img.get_fdata()
            if len(original_data.shape) == 4:
                original_data = original_data[:, :, :, 0]
            brain_nii = nib.Nifti1Image(original_data.astype(np.float32), affine)
            nib.save(brain_nii, brain_path)

        # Create overlay (skull-stripped brain + heatmap blend)
        brain_data = nib.load(brain_path).get_fdata()
        if len(brain_data.shape) == 4:
            brain_data = brain_data[:, :, :, 0]
        brain_norm = (brain_data - brain_data.min()) / (brain_data.max() - brain_data.min() + 1e-8)
        overlay = 0.7 * brain_norm + 0.3 * cam_original
        overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-8)

        overlay_path = os.path.join(output_dir, 'overlay.nii.gz')
        overlay_nii = nib.Nifti1Image(overlay.astype(np.float32), affine)
        nib.save(overlay_nii, overlay_path)

        return {
            'heatmap_nifti': heatmap_path,
            'brain_nifti': brain_path,
            'overlay_nifti': overlay_path,
            'stage': 'stage3' if is_dementia else 'stage1',
            'target_class': target_class,
            'method': 'gradcam_pp'
        }

    # ==================== SHAP EXPLAINABILITY ====================

    def _create_stage_predictor(
        self,
        model: ADRDModel,
        target_label: Optional[str],
        fixed_mri_array: np.ndarray
    ):
        """
        Create a prediction function wrapper for SHAP.

        The MRI array is held fixed -- SHAP only permutes the 187 tabular features.

        Args:
            model: The ADRDModel to wrap
            target_label: Specific label to predict probability for (None = all)
            fixed_mri_array: The patient's preprocessed MRI array (1, 128, 128, 128)

        Returns:
            Callable that takes numpy array and returns predictions
        """
        feature_cols = list(model.src_modalities.keys())
        # Separate tabular vs image columns
        tabular_cols = [c for c in feature_cols if not c.startswith('img_')]

        def predict_fn(X: np.ndarray) -> np.ndarray:
            results = []
            for row in X:
                x = {}
                tab_idx = 0
                for col in feature_cols:
                    if col.startswith('img_'):
                        x[col] = fixed_mri_array
                    else:
                        val = row[tab_idx]
                        tab_idx += 1
                        if np.isnan(val) or val in [-4, 8, 9, 88, 99, 888, 999, 8888, 9999]:
                            x[col] = None
                        else:
                            x[col] = float(val)

                with torch.no_grad():
                    # predict_proba returns (logits_list, proba_list) tuple
                    _, proba_list = model.predict_proba([x], _batch_size=1)

                if target_label:
                    prob_val = float(proba_list[0].get(target_label, 0.5))
                    results.append(prob_val)
                else:
                    probs = []
                    for label in model.tgt_modalities.keys():
                        probs.append(float(proba_list[0].get(label, 0.5)))
                    results.append(probs)

            return np.array(results)

        return predict_fn, tabular_cols

    def _create_background_data(self, n_tabular_features: int, n_samples: int = 50) -> np.ndarray:
        """Create zeros background data for SHAP (represents missing/baseline)."""
        return np.zeros((n_samples, n_tabular_features))

    def compute_tabular_shap(
        self,
        mri_array: np.ndarray,
        tabular_features: Dict[str, Any],
        stage: str,
        n_background_samples: int = 50,
        nsamples: int = 100
    ) -> Dict:
        """
        Compute SHAP values for the 187 tabular features (MRI held fixed).

        Args:
            mri_array: Preprocessed MRI array (1, 128, 128, 128)
            tabular_features: Dict of 187 tabular feature values
            stage: 'stage1', 'stage2', or 'stage3'
            n_background_samples: Number of background samples for SHAP
            nsamples: Number of SHAP evaluation samples

        Returns:
            Dict with SHAP values and feature names
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not installed. Install with: pip install shap")

        if stage == 'stage1':
            model = self.stage1_model
            target_label = 'DE'
        elif stage == 'stage2':
            model = self.stage2_model
            target_label = None
        elif stage == 'stage3':
            model = self.stage3_model
            target_label = None
        else:
            raise ValueError(f"Unknown stage: {stage}")

        predict_fn, tabular_cols = self._create_stage_predictor(model, target_label, mri_array)

        # Create background and sample arrays (tabular features only)
        background = self._create_background_data(len(tabular_cols), n_background_samples)

        sample_values = []
        for col in tabular_cols:
            val = tabular_features.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                sample_values.append(np.nan)
            else:
                sample_values.append(float(val))
        sample_array = np.array(sample_values).reshape(1, -1)

        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(sample_array, nsamples=nsamples)

        if target_label:
            shap_dict = {
                'feature_names': tabular_cols,
                'shap_values': shap_values[0].tolist(),
                'base_value': float(explainer.expected_value),
                'target': target_label
            }
        else:
            labels = list(model.tgt_modalities.keys())
            shap_dict = {
                'feature_names': tabular_cols,
                'shap_values': {
                    label: shap_values[i][0].tolist()
                    for i, label in enumerate(labels)
                },
                'base_values': {
                    label: float(explainer.expected_value[i])
                    for i, label in enumerate(labels)
                },
                'labels': labels
            }

        return shap_dict

    def compute_mri_importance(self) -> float:
        """
        Compute scalar MRI importance from last GradCAM++ activation.

        Returns mean attribution over brain voxels. Call after _generate_heatmap().
        """
        cam_masked = getattr(self, '_last_cam_masked', None)
        brain_mask = getattr(self, '_last_brain_mask_orig', None)

        if cam_masked is None:
            return 0.0

        if brain_mask is not None:
            brain_values = cam_masked[brain_mask]
            if len(brain_values) > 0:
                return float(brain_values.mean())

        return float(cam_masked.mean())

    def get_top_shap_features(
        self,
        shap_result: Dict,
        n_top: int = 10,
        target_label: Optional[str] = None
    ) -> List[Dict]:
        """
        Extract top contributing features from SHAP results.

        Args:
            shap_result: Output from compute_tabular_shap()
            n_top: Number of top features to return
            target_label: For multi-class, specify which label

        Returns:
            List of dicts with feature name, SHAP value, direction, importance
        """
        feature_names = shap_result['feature_names']

        if 'target' in shap_result:
            shap_values = np.array(shap_result['shap_values'])
        else:
            if target_label is None:
                target_label = shap_result['labels'][0]
            shap_values = np.array(shap_result['shap_values'][target_label])

        sorted_indices = np.argsort(np.abs(shap_values))[::-1][:n_top]

        top_features = []
        for idx in sorted_indices:
            val = shap_values[idx]
            top_features.append({
                'feature': feature_names[idx],
                'description': self._get_feature_description(feature_names[idx]),
                'shap_value': float(val),
                'direction': 'increases' if val > 0 else 'decreases',
                'importance': float(abs(val))
            })

        return top_features

    def predict_with_explanations(
        self,
        mri_path: str,
        features: Dict[str, Any],
        output_dir: str,
        n_top_features: int = 20,
        compute_shap: bool = True,
        nsamples: int = 100
    ) -> Dict[str, Any]:
        """
        Full inference with tabular SHAP + MRI heatmap.

        Args:
            mri_path: Path to .nii or .nii.gz MRI scan
            features: Dict of 187 tabular features
            output_dir: Directory for NIfTI heatmap outputs
            n_top_features: Number of top SHAP features to include
            compute_shap: Whether to compute SHAP values (slow, ~3-5 min)
            nsamples: Number of SHAP evaluation samples

        Returns:
            Dict with predictions, SHAP explanations, heatmap paths
        """
        # Run base prediction with heatmap
        result = self.predict(
            mri_path=mri_path,
            features=features,
            generate_heatmap=True,
            output_dir=output_dir
        )

        # Compute MRI importance from GradCAM++ activation
        mri_importance = self.compute_mri_importance()

        is_dementia = result['stage1']['prediction'] == 'DE'

        # Preprocess MRI for SHAP (reuse from predict if available)
        mri_array, _ = self.preprocessor.preprocess(mri_path)

        if compute_shap and SHAP_AVAILABLE:
            all_top_features = []

            # Stage 1 SHAP (always computed)
            try:
                print("[SHAP] Computing Stage 1 explanations...")
                stage1_shap = self.compute_tabular_shap(
                    mri_array, features, 'stage1', nsamples=nsamples
                )
                stage1_top = self.get_top_shap_features(stage1_shap, n_top_features)
                result['stage1']['explanation'] = {
                    'top_features': stage1_top,
                    'mri_importance': mri_importance
                }
                all_top_features.extend(stage1_top[:5])
            except Exception as e:
                warnings.warn(f"Stage 1 SHAP failed: {e}")
                result['stage1']['explanation'] = {'error': str(e)}

            if is_dementia and result['stage3'] is not None:
                try:
                    print("[SHAP] Computing Stage 3 explanations...")
                    stage3_shap = self.compute_tabular_shap(
                        mri_array, features, 'stage3', nsamples=nsamples
                    )
                    top_subtype = result['stage3']['prediction']
                    stage3_top = self.get_top_shap_features(
                        stage3_shap, n_top_features, target_label=top_subtype
                    )
                    result['stage3']['explanation'] = {
                        'top_features': stage3_top,
                        'mri_importance': mri_importance
                    }
                    all_top_features.extend(stage3_top[:5])
                except Exception as e:
                    warnings.warn(f"Stage 3 SHAP failed: {e}")
                    result['stage3']['explanation'] = {'error': str(e)}

            elif not is_dementia and result['stage2'] is not None:
                try:
                    print("[SHAP] Computing Stage 2 explanations...")
                    stage2_shap = self.compute_tabular_shap(
                        mri_array, features, 'stage2', nsamples=nsamples
                    )
                    top_class = result['stage2']['prediction']
                    stage2_top = self.get_top_shap_features(
                        stage2_shap, n_top_features, target_label=top_class
                    )
                    result['stage2']['explanation'] = {
                        'top_features': stage2_top,
                        'mri_importance': mri_importance
                    }
                    all_top_features.extend(stage2_top[:5])
                except Exception as e:
                    warnings.warn(f"Stage 2 SHAP failed: {e}")
                    result['stage2']['explanation'] = {'error': str(e)}

            # Aggregate top features across stages, deduplicate
            seen = set()
            unique_features = []
            for f in sorted(all_top_features, key=lambda x: x['importance'], reverse=True):
                if f['feature'] not in seen:
                    seen.add(f['feature'])
                    unique_features.append(f)

            # Build the active stage SHAP result for feature_importance
            active_shap = stage1_shap if 'stage1_shap' not in dir() else stage1_shap
            try:
                active_shap = stage1_shap
            except NameError:
                active_shap = None

            result['feature_importance'] = {
                'tabular_shap': active_shap,
                'mri_importance': mri_importance,
                'top_features': unique_features[:n_top_features]
            }

        elif compute_shap and not SHAP_AVAILABLE:
            warnings.warn("SHAP not installed. Running without tabular explanations.")
            result['feature_importance'] = {
                'tabular_shap': None,
                'mri_importance': mri_importance,
                'top_features': []
            }
        else:
            result['feature_importance'] = {
                'tabular_shap': None,
                'mri_importance': mri_importance,
                'top_features': []
            }

        return result

    def save_result(self, result: Dict, output_path: str):
        """Save prediction result to JSON."""
        # Convert numpy types for JSON serialization
        def _convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj

        def _deep_convert(d):
            if isinstance(d, dict):
                return {k: _deep_convert(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [_deep_convert(v) for v in d]
            else:
                return _convert(d)

        with open(output_path, 'w') as f:
            json.dump(_deep_convert(result), f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='MRI + Tabular Hierarchical Inference v2.0')
    parser.add_argument('--mri_path', type=str, required=True,
                        help='Path to MRI scan (.nii or .nii.gz)')
    parser.add_argument('--features_json', type=str, required=True,
                        help='Path to tabular features JSON')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--generate_heatmap', action='store_true',
                        help='Generate GradCAM++ heatmap')
    parser.add_argument('--with_shap', action='store_true',
                        help='Compute SHAP explanations (slow, ~3-5 min)')
    parser.add_argument('--stage1_checkpoint', type=str,
                        default=os.path.join(PACKAGE_DIR, 'models', 'stage1_de_vs_non_de.pt'))
    parser.add_argument('--stage2_checkpoint', type=str,
                        default=os.path.join(PACKAGE_DIR, 'models', 'stage2_nc_mci_imci.pt'))
    parser.add_argument('--stage3_checkpoint', type=str,
                        default=os.path.join(PACKAGE_DIR, 'models', 'stage3_dementia_subtypes.pt'))
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Load tabular features
    with open(args.features_json, 'r') as f:
        features = json.load(f)

    # Initialize pipeline
    pipeline = HierarchicalInferencePipeline(
        stage1_checkpoint=args.stage1_checkpoint,
        stage2_checkpoint=args.stage2_checkpoint,
        stage3_checkpoint=args.stage3_checkpoint,
        device=args.device
    )

    # Run prediction
    os.makedirs(args.output_dir, exist_ok=True)

    if args.with_shap:
        result = pipeline.predict_with_explanations(
            mri_path=args.mri_path,
            features=features,
            output_dir=args.output_dir
        )
    else:
        result = pipeline.predict(
            mri_path=args.mri_path,
            features=features,
            generate_heatmap=args.generate_heatmap,
            output_dir=args.output_dir
        )

    # Save result
    result_path = os.path.join(args.output_dir, 'prediction_result.json')
    pipeline.save_result(result, result_path)

    # Print summary
    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"Stage 1 (DE vs Non-DE): {result['stage1']['prediction']}")
    print(f"  DE Probability: {result['stage1']['de_probability']:.4f}")

    if result['stage2']:
        print(f"\nStage 2 (NC/MCI/iMCI): {result['stage2']['prediction']}")
        for k, v in result['stage2']['probabilities'].items():
            print(f"  {k}: {v:.4f}")

    if result['stage3']:
        print(f"\nStage 3 (Dementia Subtype): {result['stage3']['prediction']}")
        for k, v in sorted(result['stage3']['probabilities'].items(), key=lambda x: -x[1])[:5]:
            print(f"  {k}: {v:.4f}")

    print(f"\nFINAL DIAGNOSIS: {result['final_diagnosis']}")

    if result.get('heatmaps'):
        print(f"\nHeatmaps saved to:")
        for key, path in result['heatmaps'].items():
            if isinstance(path, str) and path.endswith('.nii.gz'):
                print(f"  {key}: {path}")

    if result.get('feature_importance', {}).get('top_features'):
        print(f"\nTop contributing features:")
        for feat in result['feature_importance']['top_features'][:10]:
            print(f"  {feat['feature']}: {feat['shap_value']:+.4f} ({feat['direction']})")
        print(f"  MRI importance: {result['feature_importance']['mri_importance']:.4f}")

    print(f"\nFull result saved to: {result_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
