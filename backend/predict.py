import pickle
import numpy as np
import pandas as pd
import shap

def load_model():
    with open("models/xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return model, feature_cols

def predict_rul(model, feature_cols, sensor_values: dict):
    input_df   = pd.DataFrame([sensor_values])[feature_cols]
    prediction = model.predict(input_df)[0]
    return round(float(np.clip(prediction, 0, 125)), 2)

def get_shap_values(model, feature_cols, sensor_values: dict):
    input_df  = pd.DataFrame([sensor_values])[feature_cols]
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(input_df)
    shap_df   = pd.DataFrame({
        "Feature"   : feature_cols,
        "SHAP Value": shap_vals[0],
        "Direction" : ["Increases RUL" if v > 0 else "Decreases RUL"
                       for v in shap_vals[0]]
    }).sort_values("SHAP Value", key=abs, ascending=False)
    return shap_df

def get_global_shap(model, feature_cols, X_test):
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test)
    shap_df   = pd.DataFrame({
        "Feature"        : feature_cols,
        "Mean SHAP Value": np.abs(shap_vals).mean(axis=0)
    }).sort_values("Mean SHAP Value", ascending=False)
    return shap_df

def get_risk_level(rul):
    if rul <= 30:
        return "CRITICAL", "#ef4444", "Immediate maintenance required!"
    elif rul <= 70:
        return "WARNING",  "#f59e0b", "Schedule maintenance soon."
    else:
        return "HEALTHY",  "#22c55e", "Engine operating normally."
