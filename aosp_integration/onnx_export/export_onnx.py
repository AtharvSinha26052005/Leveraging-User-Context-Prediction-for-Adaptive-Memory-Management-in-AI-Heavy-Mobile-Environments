"""
export_onnx.py — Export XGBoost model to ONNX format for C++ inference

Usage:
    python aosp_integration/onnx_export/export_onnx.py

Output:
    aosp_integration/onnx_export/xgboost_phase2.onnx

Requirements:
    pip install onnx onnxmltools skl2onnx

This script converts the trained XGBoost model to ONNX format so it can be
loaded by ONNX Runtime in the C++ AOSP integration or any other runtime.
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import xgboost as xgb
    import numpy as np
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloat
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("[WARNING] onnxmltools / skl2onnx not installed.")
    print("  Install with: pip install onnxmltools skl2onnx onnx")


def export_model():
    model_path = "models_trained/xgboost_phase2.json"
    output_path = "aosp_integration/onnx_export/xgboost_phase2.onnx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.isfile(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        print("  Run python scripts/run_phase2.py first.")
        return

    print(f"Loading XGBoost model from {model_path}...")
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # Get feature dimension from phase2 metrics
    metrics_path = "results/phase2_metrics.json"
    if os.path.isfile(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        n_features = metrics.get("total_features", 46)
        n_classes  = metrics.get("n_classes", 28)
    else:
        n_features = 46
        n_classes  = 28

    print(f"  Features: {n_features}")
    print(f"  Classes:  {n_classes}")

    if not ONNX_AVAILABLE:
        print("\n[SKIPPING] ONNX export libraries not installed.")
        print("  The model is saved as JSON at:", model_path)
        print("  Once onnxmltools is installed, re-run this script.")
        _write_model_info(model, n_features, n_classes, output_path)
        return

    # Export to ONNX
    print(f"\nExporting to ONNX...")
    initial_type = [("float_input", OnnxFloat([None, n_features]))]
    onnx_model = convert_xgboost(model, initial_types=initial_type)

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    file_size_kb = os.path.getsize(output_path) / 1024
    print(f"  Saved to: {output_path}")
    print(f"  File size: {file_size_kb:.1f} KB")
    print(f"\n[OK] ONNX model ready for C++ inference via ONNX Runtime.")
    print(f"     Load in C++ with: Ort::Session session(env, L\"{output_path}\", opts);")


def _write_model_info(model, n_features, n_classes, output_path):
    """Write model info JSON when ONNX not available."""
    info = {
        "model_type": "XGBClassifier",
        "n_features": n_features,
        "n_classes": n_classes,
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "feature_layout": {
            "base_features_0_18": "FeatureEngine 19-dim (hour, battery, RAM, etc.)",
            "app2vec_19_34": "App2Vec PCA 8-dim x 2 apps",
            "prefixspan_35_45": "PrefixSpan 10 binary flags + 1 confidence"
        },
        "onnx_export_path": output_path,
        "onnx_status": "pending_library_install",
        "install_command": "pip install onnxmltools skl2onnx onnx"
    }
    info_path = output_path.replace(".onnx", "_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Model info written to: {info_path}")


if __name__ == "__main__":
    export_model()
