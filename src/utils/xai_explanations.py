import shap
import numpy as np

# This module provides XAI (SHAP) explanations for model predictions

def explain_prediction(model, X, feature_names=None):
    """
    Generate SHAP values for a single prediction.
    Args:
        model: Trained model (sklearn, xgboost, keras, etc.)
        X: Input sample (1D or 2D array)
        feature_names: List of feature names (optional)
    Returns:
        shap_values: SHAP values for the input
        expected_value: Model expected value
    """
    # Try TreeExplainer for tree models, else KernelExplainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        expected_value = explainer.expected_value
    except Exception:
        explainer = shap.KernelExplainer(model.predict, shap.sample(X, 100))
        shap_values = explainer.shap_values(X)
        expected_value = explainer.expected_value
    return shap_values, expected_value

# Example usage (to be called from Flask route):
# shap_values, expected_value = explain_prediction(model, X, feature_names)
