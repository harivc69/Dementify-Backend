#!/usr/bin/env python3
"""
Simple REST API Server for Hierarchical Cognitive Classifier

This provides a simple HTTP API that a frontend can call.
The frontend engineer can use this as a reference or run it directly.

Endpoints:
    GET  /health          - Health check
    GET  /features        - Get list of expected features
    POST /predict         - Run inference on one or more samples
    POST /predict/file    - Upload CSV/JSON file for batch inference

Usage:
    python api_server.py --port 8080

Requirements:
    pip install flask flask-cors
"""

import sys
import json
import argparse
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hierarchical_classifier import HierarchicalCognitiveClassifier

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Global classifier instance (loaded once at startup)
classifier = None


def get_classifier():
    """Get or initialize the classifier."""
    global classifier
    if classifier is None:
        print("Loading models...")
        classifier = HierarchicalCognitiveClassifier()
    return classifier


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': classifier is not None
    })


@app.route('/features', methods=['GET'])
def get_features():
    """Get the list of expected input features."""
    clf = get_classifier()
    features = clf.get_feature_list()

    return jsonify({
        'total_features': len(features),
        'features': features,
        'feature_info': clf.get_feature_info()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Run inference on one or more samples.

    Request body (JSON):
        Single sample: {"his_NACCAGE": 75, "his_SEX": 2, ...}
        Multiple samples: [{"his_NACCAGE": 75, ...}, {"his_NACCAGE": 80, ...}]

    Response:
        {
            "success": true,
            "count": 1,
            "results": [
                {
                    "sample_id": 0,
                    "stage1": {...},
                    "stage2": {...} or null,
                    "stage3": {...} or null,
                    "summary": "..."
                }
            ]
        }
    """
    try:
        data = request.get_json()

        if data is None:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400

        clf = get_classifier()
        results = clf.predict(data)

        return jsonify({
            'success': True,
            'count': len(results),
            'results': results
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict/file', methods=['POST'])
def predict_file():
    """
    Run inference on uploaded CSV or JSON file.

    Request:
        Content-Type: multipart/form-data
        file: CSV or JSON file

    Response: Same as /predict
    """
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400

        file = request.files['file']
        filename = file.filename.lower()

        if filename.endswith('.csv'):
            import pandas as pd
            import io
            content = file.read().decode('utf-8')
            data = pd.read_csv(io.StringIO(content))
        elif filename.endswith('.json'):
            data = json.load(file)
        else:
            return jsonify({
                'success': False,
                'error': 'Unsupported file format. Use CSV or JSON.'
            }), 400

        clf = get_classifier()
        results = clf.predict(data)

        return jsonify({
            'success': True,
            'count': len(results),
            'results': results
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict/manual', methods=['POST'])
def predict_manual():
    """
    Run inference with manual feature entry.
    Missing features will be automatically imputed.

    Request body (JSON):
        {
            "features": {
                "his_NACCAGE": 75,
                "his_SEX": 2,
                "bat_NACCMMSE": 22
                // Only include known features, rest will be imputed
            }
        }

    Response: Same as /predict
    """
    try:
        data = request.get_json()

        if data is None or 'features' not in data:
            return jsonify({
                'success': False,
                'error': 'Request must include "features" object'
            }), 400

        clf = get_classifier()

        # Get all feature names
        all_features = clf.get_feature_list()

        # Create input with missing values for unknown features
        input_data = {}
        provided_features = data['features']

        for feat in all_features:
            if feat in provided_features:
                input_data[feat] = provided_features[feat]
            else:
                input_data[feat] = -4  # Missing value code

        results = clf.predict(input_data)

        return jsonify({
            'success': True,
            'count': 1,
            'provided_features': len(provided_features),
            'total_features': len(all_features),
            'results': results
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def main():
    parser = argparse.ArgumentParser(description='Cognitive Classifier API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    # Pre-load the classifier
    print("=" * 70)
    print("Hierarchical Cognitive Classifier - API Server")
    print("=" * 70)
    get_classifier()

    print(f"\nStarting server on http://{args.host}:{args.port}")
    print("\nEndpoints:")
    print("  GET  /health        - Health check")
    print("  GET  /features      - List expected features")
    print("  POST /predict       - Predict from JSON body")
    print("  POST /predict/file  - Predict from uploaded file")
    print("  POST /predict/manual - Predict with partial features")
    print("\n" + "=" * 70)

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
