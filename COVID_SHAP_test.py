# ================================================================
# XGBoost + SHAP + p-value Analysis (Binary outcome)
# Safe, stable version – November 2025
# ================================================================

import pandas as pd;import numpy as np;import shap
import xgboost as xgb;import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score;from scipy import stats
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# === 1. Load Excel file ===
excel_path = r"C:\Users\fadhil abidi\Desktop\COVID19_paper\COVID19_Patient_Sample_Imputed_200.xlsx"
df = pd.read_excel(excel_path)

# === 2. Clean column names ===
df.columns = df.columns.str.strip().str.replace(r"\s+", "_", regex=True)

# === 3. Detect 'outcome' column automatically (case-insensitive) ===
outcome_col = None
for c in df.columns:
    if "outcome" in c.lower():
        outcome_col = c
        break

if outcome_col is None:
    raise ValueError("❌ No column containing 'outcome' found in dataset.")
else:
    print(f"✅ Using outcome column: {outcome_col}")

# === 4. Keep only rows where outcome is 0 or 1 ===
df = df[df[outcome_col].isin([0, 1])].copy()

# === 5. Separate features and target ===
y = df[outcome_col];X = df.drop(columns=[outcome_col])

# === 6. Ensure WBC exists ===
if "WBC" not in X.columns:
    print("⚠️ Warning: 'WBC' feature not found in dataset.")
    print("Available columns:")
    print(X.columns.tolist())

# === 7. Encode categorical columns ===
for col in X.select_dtypes(include=["object", "category"]).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# === 8. Split data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# === 9. Train XGBoost binary classifier ===
model = xgb.XGBClassifier(
    objective="binary:logistic",
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42)
model.fit(X_train, y_train)

# === 10. Evaluate model ===
y_pred = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print(f"✅ Model AUC: {auc:.3f}")

# === 11. Compute SHAP values (Safe KernelExplainer version) ===
print("⚙️ Computing SHAP values safely using KernelExplainer...")

# Use small random subset for SHAP background (speed)
background = shap.sample(X_train, 100, random_state=42)

# Safe callable wrapper to avoid XGBoost attribute errors
def model_predict(X):
    return model.predict_proba(X)[:, 1]

explainer = shap.KernelExplainer(model_predict, background)

# Compute SHAP values for a representative subset
X_sample = shap.sample(X_train, 300, random_state=42)
shap_values = explainer.shap_values(X_sample)

# Handle binary/multiclass outputs
if isinstance(shap_values, list) and len(shap_values) > 1:
    shap_array = np.array(shap_values[1])
else:
    shap_array = np.array(shap_values)

print("✅ SHAP values computed successfully.")

# === 12. SHAP feature importance ===
shap_importance = np.abs(shap_array).mean(axis=0)
shap_df = pd.DataFrame({
    "Feature": X.columns,
    "Mean(|SHAP|)": shap_importance
}).sort_values("Mean(|SHAP|)", ascending=False)

# === 13. Compute p-values (Spearman correlation) ===
p_values = []
for feature in X.columns:
    try:
        _, pval = stats.spearmanr(X[feature], y)
    except Exception:
        pval = np.nan
    p_values.append(pval)

shap_df["p_value"] = p_values
shap_df["Significant"] = shap_df["p_value"] < 0.05

# === 14. Save results to Excel ===
output_excel = r"C:\Users\fadhil abidi\Desktop\COVID19_paper\XGB_SHAP_Results.xlsx"
shap_df.to_excel(output_excel, index=False)
print(f"✅ Results saved to: {output_excel}")

# === 15. Plot SHAP summary (all features) ===
plt.figure()
shap.summary_plot(shap_array, X_sample, show=False)
plt.title("SHAP Summary (All Features)")
plt.tight_layout()
plt.savefig(r"C:\Users\fadhil abidi\Desktop\COVID19_paper\xgb_shap_beeswarm_all.png", dpi=300)
plt.close()

# === 16. Plot SHAP summary (significant features only) ===
sig_features = shap_df.loc[shap_df["Significant"], "Feature"].tolist()
if sig_features:
    idx = [X.columns.get_loc(c) for c in sig_features if c in X.columns]
    if idx:
        plt.figure()
        shap.summary_plot(shap_array[:, idx], X_sample[sig_features], show=False)
        plt.title("SHAP Summary (Significant Features, p<0.05)")
        plt.tight_layout()
        plt.savefig(r"C:\Users\fadhil abidi\Desktop\COVID19_paper\xgb_shap_beeswarm_significant.png", dpi=300)
        plt.close()
        print("✅ SHAP significant features plot saved.")
else:
    print("⚠️ No significant features (p < 0.05).")

# === 17. Print top 10 features ===
print("\n🔹 Top 10 features by SHAP importance:")
print(shap_df.head(10).to_string(index=False))

print("\n✅ Analysis complete.")
