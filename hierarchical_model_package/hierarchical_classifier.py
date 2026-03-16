#!/usr/bin/env python3
"""
Hierarchical Cognitive Impairment Classifier

A unified inference pipeline for differential diagnosis of cognitive impairment
using a 3-stage hierarchical transformer model.

Hierarchy:
    Stage 1: Dementia (DE) vs Non-Dementia
        |
        +-- If Non-Dementia --> Stage 2: NC vs MCI vs iMCI
        |
        +-- If Dementia --> Stage 3: 10 Dementia Subtypes
                           (AD, LBD, VD, FTD, PRD, NPH, SEF, PSY, TBI, ODE)

Features:
    - Handles missing data through learned imputation
    - Supports batch inference from CSV or JSON
    - Returns hierarchical predictions with confidence scores

Authors: UIUC Research Team
License: Research Use Only
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import warnings
warnings.filterwarnings('ignore')

# SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not installed. Install with: pip install shap")

# Add the nmed2024 package to path if running standalone
PACKAGE_DIR = Path(__file__).parent
NMED_PATH = PACKAGE_DIR.parent / "nmed2024"
if NMED_PATH.exists():
    sys.path.insert(0, str(NMED_PATH))
    sys.path.insert(0, str(NMED_PATH / "dev"))


class HierarchicalCognitiveClassifier:
    """
    Hierarchical classifier for cognitive impairment diagnosis.

    This classifier implements a 3-stage hierarchical pipeline:
    1. Stage 1: Dementia vs Non-Dementia (binary)
    2. Stage 2: NC vs MCI vs iMCI (for non-dementia cases)
    3. Stage 3: 10 Dementia Subtypes (for dementia cases)

    The model handles missing data through learned imputation - features with
    missing values are automatically imputed using learned embeddings.

    Attributes:
        device (str): 'cuda' or 'cpu'
        stage1_model: Binary classifier for DE vs Non-DE
        stage2_model: 3-way classifier for NC/MCI/iMCI
        stage3_model: 10-way classifier for dementia subtypes
    """

    # Class labels for each stage
    STAGE1_LABELS = ['Non-Dementia', 'Dementia']
    STAGE2_LABELS = ['NC', 'MCI', 'iMCI']
    STAGE3_LABELS = ['AD', 'LBD', 'VD', 'FTD', 'PRD', 'NPH', 'SEF', 'PSY', 'TBI', 'ODE']

    # Full names for dementia subtypes
    SUBTYPE_NAMES = {
        'AD': "Alzheimer's Disease",
        'LBD': 'Lewy Body Dementia',
        'VD': 'Vascular Dementia',
        'FTD': 'Frontotemporal Dementia',
        'PRD': 'Prion Disease (CJD)',
        'NPH': 'Normal Pressure Hydrocephalus',
        'SEF': 'Systemic/Environmental Factors',
        'PSY': 'Psychiatric Causes',
        'TBI': 'Traumatic Brain Injury',
        'ODE': 'Other Dementia Etiologies'
    }

    def __init__(
        self,
        models_dir: Optional[str] = None,
        configs_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the hierarchical classifier.

        Args:
            models_dir: Directory containing model checkpoint files.
                       Defaults to 'models/' in package directory.
            configs_dir: Directory containing TOML config files.
                        Defaults to 'configs/' in package directory.
            device: 'cuda' or 'cpu'. Auto-detected if None.
        """
        # Set directories
        self.package_dir = PACKAGE_DIR
        self.models_dir = Path(models_dir) if models_dir else self.package_dir / "models"
        self.configs_dir = Path(configs_dir) if configs_dir else self.package_dir / "configs"

        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"[HierarchicalClassifier] Initializing on device: {self.device}")

        # Load models
        self._load_models()

        print("[HierarchicalClassifier] All models loaded successfully!")

    def _load_models(self):
        """Load all three stage models."""
        from adrd.model import ADRDModel
        from data.dataset_csv import CSVDataset

        # Stage 1: DE vs Non-DE
        print("[HierarchicalClassifier] Loading Stage 1 model (DE vs Non-DE)...")
        stage1_config = self.configs_dir / "stage1_de_config.toml"
        stage1_ckpt = self.models_dir / "stage1_de_vs_nonde.pt"
        self.stage1_model = self._load_single_model(stage1_config, stage1_ckpt)

        # Stage 2: NC vs MCI vs iMCI
        print("[HierarchicalClassifier] Loading Stage 2 model (NC/MCI/iMCI)...")
        stage2_config = self.configs_dir / "stage2_3way_config.toml"
        stage2_ckpt = self.models_dir / "stage2_nc_mci_imci.pt"
        self.stage2_model = self._load_single_model(stage2_config, stage2_ckpt)

        # Stage 3: 10 Dementia Subtypes
        print("[HierarchicalClassifier] Loading Stage 3 model (10 Dementia Subtypes)...")
        stage3_config = self.configs_dir / "stage3_10class_config.toml"
        stage3_ckpt = self.models_dir / "stage3_dementia_subtypes.pt"
        self.stage3_model = self._load_single_model(stage3_config, stage3_ckpt)

    def _load_single_model(self, config_path: Path, ckpt_path: Path) -> 'ADRDModel':
        """Load a single ADRDModel from checkpoint."""
        import toml
        from adrd.model import ADRDModel

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # Parse config to get modalities
        config = toml.load(config_path)

        # Build src_modalities (features) and tgt_modalities (labels) from config
        src_modalities = {}
        tgt_modalities = {}

        for key, val in config.get('feature', {}).items():
            src_modalities[key] = val

        for key, val in config.get('label', {}).items():
            tgt_modalities[key] = val

        # Compute label fractions (use uniform for inference)
        label_fractions = {k: 1.0 / len(tgt_modalities) for k in tgt_modalities.keys()}

        # Initialize model
        model = ADRDModel(
            src_modalities=src_modalities,
            tgt_modalities=tgt_modalities,
            label_fractions=label_fractions,
            d_model=128,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            device=self.device,
            img_net='NonImg',
            _dataloader_num_workers=0
        )

        # Load checkpoint using the model's built-in load method
        device_str = 'cuda' if self.device == torch.device('cuda') else 'cpu'
        model.load(str(ckpt_path), map_location=device_str)

        return model

    def _prepare_input(self, data: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """
        Prepare input data for model inference.

        Handles missing data by setting mask to 0 for missing values.
        The model will use learned imputation for these values.

        Args:
            data: DataFrame with feature columns

        Returns:
            Tuple of (features_list, labels_list) for model input
        """
        features = []
        labels = []

        for idx, row in data.iterrows():
            feat_dict = {}
            label_dict = {}

            for col in data.columns:
                val = row[col]
                # Handle missing values
                if pd.isna(val) or val in [-4, 8, 9, 88, 99, 888, 999, 8888, 9999]:
                    feat_dict[col] = (np.nan, 0)  # (value, mask=0 for missing)
                else:
                    feat_dict[col] = (float(val), 1)  # (value, mask=1 for valid)

            features.append(feat_dict)
            labels.append(label_dict)

        return features, labels

    def predict(
        self,
        data: Union[pd.DataFrame, Dict, List[Dict], str],
        return_probabilities: bool = True
    ) -> List[Dict]:
        """
        Run hierarchical inference on input data.

        Args:
            data: Input data in one of the following formats:
                  - pandas DataFrame with feature columns
                  - Dict with feature values (single sample)
                  - List of Dicts (multiple samples)
                  - Path to CSV or JSON file
            return_probabilities: Whether to include probability scores

        Returns:
            List of prediction dictionaries, one per sample:
            {
                'sample_id': int,
                'stage1': {
                    'prediction': 'Dementia' or 'Non-Dementia',
                    'confidence': float,
                    'probabilities': {'Non-Dementia': float, 'Dementia': float}
                },
                'stage2': {...} or None,  # Only if Non-Dementia
                'stage3': {...} or None,  # Only if Dementia
                'summary': str  # Human-readable summary
            }
        """
        # Load data if path provided
        if isinstance(data, str):
            data = self._load_data_file(data)
        elif isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, list):
            data = pd.DataFrame(data)

        results = []

        for idx in range(len(data)):
            sample = data.iloc[[idx]]
            result = self._predict_single(sample, idx, return_probabilities)
            results.append(result)

        return results

    def _predict_single(
        self,
        sample: pd.DataFrame,
        sample_id: int,
        return_probabilities: bool
    ) -> Dict:
        """Run hierarchical inference on a single sample."""
        result = {
            'sample_id': sample_id,
            'stage1': None,
            'stage2': None,
            'stage3': None,
            'summary': ''
        }

        # Stage 1: DE vs Non-DE
        stage1_probs = self._run_stage_inference(self.stage1_model, sample)
        de_prob = stage1_probs.get('DE', 0.5)
        is_dementia = de_prob >= 0.5

        result['stage1'] = {
            'prediction': 'Dementia' if is_dementia else 'Non-Dementia',
            'confidence': de_prob if is_dementia else (1 - de_prob),
            'de_probability': de_prob
        }
        if return_probabilities:
            result['stage1']['probabilities'] = {
                'Non-Dementia': 1 - de_prob,
                'Dementia': de_prob
            }

        if is_dementia:
            # Stage 3: Dementia Subtypes
            stage3_probs = self._run_stage_inference(self.stage3_model, sample)

            # Get top predictions
            sorted_subtypes = sorted(stage3_probs.items(), key=lambda x: x[1], reverse=True)
            top_subtype = sorted_subtypes[0][0]

            result['stage3'] = {
                'prediction': top_subtype,
                'prediction_full_name': self.SUBTYPE_NAMES.get(top_subtype, top_subtype),
                'confidence': sorted_subtypes[0][1],
                'top_3': [
                    {'subtype': s, 'name': self.SUBTYPE_NAMES.get(s, s), 'probability': p}
                    for s, p in sorted_subtypes[:3]
                ]
            }
            if return_probabilities:
                result['stage3']['all_probabilities'] = stage3_probs

            result['summary'] = (
                f"DEMENTIA (confidence: {de_prob:.1%}) - "
                f"Primary Subtype: {self.SUBTYPE_NAMES.get(top_subtype, top_subtype)} "
                f"({sorted_subtypes[0][1]:.1%})"
            )
        else:
            # Stage 2: NC vs MCI vs iMCI
            stage2_probs = self._run_stage_inference(self.stage2_model, sample)

            # Get top prediction
            sorted_classes = sorted(stage2_probs.items(), key=lambda x: x[1], reverse=True)
            top_class = sorted_classes[0][0]

            class_names = {
                'NC': 'Normal Cognition',
                'MCI': 'Mild Cognitive Impairment',
                'IMCI': 'MCI with Functional Impairment'
            }

            result['stage2'] = {
                'prediction': top_class,
                'prediction_full_name': class_names.get(top_class, top_class),
                'confidence': sorted_classes[0][1]
            }
            if return_probabilities:
                result['stage2']['probabilities'] = stage2_probs

            result['summary'] = (
                f"NON-DEMENTIA (confidence: {1-de_prob:.1%}) - "
                f"{class_names.get(top_class, top_class)} ({sorted_classes[0][1]:.1%})"
            )

        return result

    def _run_stage_inference(self, model: 'ADRDModel', sample: pd.DataFrame) -> Dict[str, float]:
        """Run inference for a single stage model."""
        # Get feature columns from model
        feature_cols = list(model.src_modalities.keys())

        # Prepare input - model expects {feature_name: value} or {feature_name: None} for missing
        x = {}
        for col in feature_cols:
            if col in sample.columns:
                val = sample[col].values[0]
                if pd.isna(val) or val in [-4, 8, 9, 88, 99, 888, 999, 8888, 9999]:
                    x[col] = None  # Missing value
                else:
                    x[col] = float(val)  # Valid value
            else:
                x[col] = None  # Column not present: treat as missing

        # Run prediction
        with torch.no_grad():
            logits, proba, preds = model.predict([x])

        # Extract probabilities from proba (which is a list of dicts)
        result = {}
        for label in model.tgt_modalities.keys():
            result[label] = float(proba[0][label])

        return result

    def _load_data_file(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV or JSON file."""
        filepath = Path(filepath)

        if filepath.suffix.lower() == '.csv':
            return pd.read_csv(filepath)
        elif filepath.suffix.lower() == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.DataFrame(data)
            else:
                return pd.DataFrame([data])
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

    def get_feature_list(self) -> List[str]:
        """Get the list of required input features (187 features)."""
        return list(self.stage1_model.src_modalities.keys())

    def get_feature_info(self) -> Dict:
        """Get detailed information about input features."""
        features = {}
        for name, info in self.stage1_model.src_modalities.items():
            features[name] = {
                'type': info.get('type', 'numerical'),
                'description': self._get_feature_description(name)
            }
        return features

    def _get_feature_description(self, name: str) -> str:
        """Get human-readable description for a feature."""
        prefixes = {
            'his_': 'Medical History',
            'med_': 'Medication',
            'ph_': 'Physical Measurement',
            'bat_': 'Cognitive Test',
            'exam_': 'Neurological Exam',
            'cvd_': 'Cardiovascular',
            'updrs_': 'UPDRS Score',
            'npiq_': 'Neuropsychiatric',
            'gds_': 'Depression Scale',
            'faq_': 'Functional Assessment'
        }

        for prefix, category in prefixes.items():
            if name.startswith(prefix):
                return f"{category}: {name[len(prefix):]}"
        return name

    # ==================== SHAP EXPLAINABILITY ====================

    def _create_stage_predictor(self, model: 'ADRDModel', target_label: Optional[str] = None):
        """
        Create a prediction function wrapper for SHAP.

        Args:
            model: The ADRDModel to wrap
            target_label: Specific label to predict probability for (optional)

        Returns:
            Callable that takes numpy array and returns predictions
        """
        feature_cols = list(model.src_modalities.keys())

        def predict_fn(X: np.ndarray) -> np.ndarray:
            """Wrapper function for SHAP - takes numpy array, returns probabilities."""
            results = []

            for row in X:
                # Convert numpy row to feature dict
                x = {}
                for i, col in enumerate(feature_cols):
                    val = row[i]
                    if np.isnan(val) or val in [-4, 8, 9, 88, 99, 888, 999, 8888, 9999]:
                        x[col] = None
                    else:
                        x[col] = float(val)

                # Run prediction
                with torch.no_grad():
                    _, proba, _ = model.predict([x])

                if target_label:
                    # Return probability for specific label
                    results.append(proba[0][target_label])
                else:
                    # Return all probabilities as array
                    probs = [proba[0][label] for label in model.tgt_modalities.keys()]
                    results.append(probs)

            return np.array(results)

        return predict_fn

    def _sample_to_array(self, sample: pd.DataFrame, model: 'ADRDModel') -> np.ndarray:
        """Convert a DataFrame sample to numpy array for SHAP."""
        feature_cols = list(model.src_modalities.keys())
        values = []

        for col in feature_cols:
            if col in sample.columns:
                val = sample[col].values[0]
                if pd.isna(val):
                    values.append(np.nan)
                else:
                    values.append(float(val))
            else:
                values.append(np.nan)

        return np.array(values).reshape(1, -1)

    def _create_background_data(self, model: 'ADRDModel', n_samples: int = 50) -> np.ndarray:
        """
        Create background data for SHAP using feature means/modes.

        For SHAP, we need a background dataset representing "typical" values.
        We use zeros as baseline (representing missing/average values).
        """
        feature_cols = list(model.src_modalities.keys())
        # Use zeros as background (represents baseline/missing values)
        background = np.zeros((n_samples, len(feature_cols)))
        return background

    def compute_shap_values(
        self,
        sample: pd.DataFrame,
        stage: str,
        n_background_samples: int = 50
    ) -> Dict:
        """
        Compute SHAP values for a specific stage.

        Args:
            sample: Single sample DataFrame
            stage: 'stage1', 'stage2', or 'stage3'
            n_background_samples: Number of background samples for SHAP

        Returns:
            Dict with SHAP values and feature names
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not installed. Install with: pip install shap")

        # Get appropriate model
        if stage == 'stage1':
            model = self.stage1_model
            target_label = 'DE'  # Probability of dementia
        elif stage == 'stage2':
            model = self.stage2_model
            target_label = None  # All classes
        elif stage == 'stage3':
            model = self.stage3_model
            target_label = None  # All classes
        else:
            raise ValueError(f"Unknown stage: {stage}")

        feature_cols = list(model.src_modalities.keys())

        # Create prediction function wrapper
        predict_fn = self._create_stage_predictor(model, target_label)

        # Create background data
        background = self._create_background_data(model, n_background_samples)

        # Convert sample to numpy array
        sample_array = self._sample_to_array(sample, model)

        # Create SHAP KernelExplainer (model-agnostic, works with any function)
        explainer = shap.KernelExplainer(predict_fn, background)

        # Compute SHAP values
        shap_values = explainer.shap_values(sample_array, nsamples=100)

        # Format results
        if target_label:
            # Single output (e.g., Stage 1 DE probability)
            shap_dict = {
                'feature_names': feature_cols,
                'shap_values': shap_values[0].tolist(),
                'base_value': float(explainer.expected_value),
                'target': target_label
            }
        else:
            # Multi-output (Stage 2 or Stage 3)
            labels = list(model.tgt_modalities.keys())
            shap_dict = {
                'feature_names': feature_cols,
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

    def get_top_shap_features(
        self,
        shap_result: Dict,
        n_top: int = 10,
        target_label: Optional[str] = None
    ) -> List[Dict]:
        """
        Extract top contributing features from SHAP results.

        Args:
            shap_result: Output from compute_shap_values()
            n_top: Number of top features to return
            target_label: For multi-class, specify which label to get features for

        Returns:
            List of dicts with feature name, SHAP value, and direction
        """
        feature_names = shap_result['feature_names']

        if 'target' in shap_result:
            # Single target (Stage 1)
            shap_values = np.array(shap_result['shap_values'])
        else:
            # Multi-target (Stage 2 or 3)
            if target_label is None:
                target_label = shap_result['labels'][0]
            shap_values = np.array(shap_result['shap_values'][target_label])

        # Get indices sorted by absolute SHAP value
        sorted_indices = np.argsort(np.abs(shap_values))[::-1][:n_top]

        top_features = []
        for idx in sorted_indices:
            val = shap_values[idx]
            top_features.append({
                'feature': feature_names[idx],
                'feature_description': self._get_feature_description(feature_names[idx]),
                'shap_value': float(val),
                'direction': 'increases' if val > 0 else 'decreases',
                'importance': float(abs(val))
            })

        return top_features

    def predict_with_explanations(
        self,
        data: Union[pd.DataFrame, Dict, List[Dict], str],
        n_top_features: int = 20,
        compute_shap: bool = True
    ) -> List[Dict]:
        """
        Run hierarchical inference with SHAP explanations.

        This method runs the full hierarchical prediction pipeline and computes
        SHAP explanations for each active stage in the hierarchy.

        For Dementia cases: Stage 1 SHAP + Stage 3 SHAP
        For Non-Dementia cases: Stage 1 SHAP + Stage 2 SHAP

        Args:
            data: Input data (DataFrame, dict, list of dicts, or file path)
            n_top_features: Number of top SHAP features to include in results (default: 20).
                           Set to 187 to get all features ranked by importance.
            compute_shap: Whether to compute SHAP values (set False for faster inference)

        Returns:
            List of prediction dictionaries with explanations:
            {
                'sample_id': int,
                'stage1': {..., 'explanation': {...}},
                'stage2': {..., 'explanation': {...}} or None,
                'stage3': {..., 'explanation': {...}} or None,
                'summary': str,
                'top_contributing_features': [...]
            }
        """
        if compute_shap and not SHAP_AVAILABLE:
            warnings.warn("SHAP not available. Running inference without explanations.")
            compute_shap = False

        # Load data if path provided
        if isinstance(data, str):
            data = self._load_data_file(data)
        elif isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, list):
            data = pd.DataFrame(data)

        results = []

        for idx in range(len(data)):
            sample = data.iloc[[idx]]

            # Get base prediction
            result = self._predict_single(sample, idx, return_probabilities=True)

            # Preserve original_id if present
            if 'original_id' in sample.columns:
                result['original_id'] = sample['original_id'].values[0]

            if compute_shap:
                print(f"[SHAP] Computing explanations for sample {idx}...")

                all_top_features = []

                # Stage 1 SHAP (always computed)
                try:
                    stage1_shap = self.compute_shap_values(sample, 'stage1')
                    stage1_top = self.get_top_shap_features(stage1_shap, n_top_features)
                    result['stage1']['explanation'] = {
                        'top_features': stage1_top,
                        'full_shap': stage1_shap
                    }
                    all_top_features.extend(stage1_top[:5])
                except Exception as e:
                    warnings.warn(f"Stage 1 SHAP failed: {e}")
                    result['stage1']['explanation'] = {'error': str(e)}

                is_dementia = result['stage1']['prediction'] == 'Dementia'

                if is_dementia and result['stage3'] is not None:
                    # Stage 3 SHAP for dementia cases
                    try:
                        stage3_shap = self.compute_shap_values(sample, 'stage3')
                        top_subtype = result['stage3']['prediction']
                        stage3_top = self.get_top_shap_features(
                            stage3_shap, n_top_features, target_label=top_subtype
                        )
                        result['stage3']['explanation'] = {
                            'top_features': stage3_top,
                            'full_shap': stage3_shap
                        }
                        all_top_features.extend(stage3_top[:5])
                    except Exception as e:
                        warnings.warn(f"Stage 3 SHAP failed: {e}")
                        result['stage3']['explanation'] = {'error': str(e)}

                elif not is_dementia and result['stage2'] is not None:
                    # Stage 2 SHAP for non-dementia cases
                    try:
                        stage2_shap = self.compute_shap_values(sample, 'stage2')
                        top_class = result['stage2']['prediction']
                        stage2_top = self.get_top_shap_features(
                            stage2_shap, n_top_features, target_label=top_class
                        )
                        result['stage2']['explanation'] = {
                            'top_features': stage2_top,
                            'full_shap': stage2_shap
                        }
                        all_top_features.extend(stage2_top[:5])
                    except Exception as e:
                        warnings.warn(f"Stage 2 SHAP failed: {e}")
                        result['stage2']['explanation'] = {'error': str(e)}

                # Aggregate top contributing features across all stages
                # Sort by importance and deduplicate
                seen = set()
                unique_features = []
                for f in sorted(all_top_features, key=lambda x: x['importance'], reverse=True):
                    if f['feature'] not in seen:
                        seen.add(f['feature'])
                        unique_features.append(f)

                result['top_contributing_features'] = unique_features[:n_top_features]

            results.append(result)

        return results


def main():
    """Example usage of the HierarchicalCognitiveClassifier."""
    print("=" * 70)
    print("Hierarchical Cognitive Impairment Classifier")
    print("=" * 70)

    # Initialize classifier
    classifier = HierarchicalCognitiveClassifier()

    # Get feature list
    features = classifier.get_feature_list()
    print(f"\nModel expects {len(features)} input features")
    print(f"First 10 features: {features[:10]}")

    print("\n" + "=" * 70)
    print("Ready for inference!")
    print("=" * 70)
    print("\nUsage:")
    print("  classifier.predict(data)  # data can be DataFrame, dict, or file path")
    print("\nExample:")
    print("  results = classifier.predict('patient_data.csv')")
    print("  for r in results:")
    print("      print(r['summary'])")


if __name__ == '__main__':
    main()
