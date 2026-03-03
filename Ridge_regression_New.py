# Ridge Logistic Regression for  Death = 1 , Discharged = 0
#1- Import Libraries and functions
import pandas as pd ;import numpy as np;import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve;from scipy.stats import norm

# 2- Read Data from Excel file
file_path = "COVID19_Patient_Sample_Imputed_200.xlsx"  # you can change to your data file
df = pd.read_excel(file_path)

# 3- Convert Categorical Predictors
df = pd.get_dummies(df, drop_first=True)

# 4- Define X and y
y = df["outcome"];X = df.drop(columns=["outcome"])

# 5- Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 6- Standardization
scaler = StandardScaler();X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7-Ridge Logistic Regression
ridge_model = LogisticRegressionCV(penalty="l2",solver="liblinear",cv=5,
    scoring="roc_auc",max_iter=3000,random_state=42)

ridge_model.fit(X_train_scaled, y_train)

# 8-Model Evaluation
y_prob = ridge_model.predict_proba(X_test_scaled)[:, 1];auc = roc_auc_score(y_test, y_prob)

print("\nAUC Score:", round(auc, 3))

# 9-ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure();plt.plot(fpr, tpr);plt.plot([0,1], [0,1])
plt.xlabel("False Positive Rate");plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Ridge Logistic Regression");plt.show()

# 10- Approximate P-values (Wald Test)
coefficients = ridge_model.coef_[0];odds_ratios = np.exp(coefficients)

X_design = np.hstack([np.ones((X_train_scaled.shape[0], 1)), X_train_scaled])
p_train = ridge_model.predict_proba(X_train_scaled)[:, 1]

W = np.diag(p_train * (1 - p_train));lambda_val = 1 / ridge_model.C_[0]

I = np.eye(X_design.shape[1]);I[0, 0] = 0

XtWX = X_design.T @ W @ X_design
ridge_matrix = XtWX + lambda_val * I

cov_matrix = np.linalg.inv(ridge_matrix)
standard_errors = np.sqrt(np.diag(cov_matrix))

coefficients_full = np.concatenate(([ridge_model.intercept_[0]], coefficients))
z_scores = coefficients_full / standard_errors
p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))

# 11- Results Table
results = pd.DataFrame({"Variable": X.columns,"Coefficient": np.round(coefficients, 3),
    "Odds Ratio": np.round(odds_ratios, 3),"Std Error": np.round(standard_errors[1:], 3),
    "z-value": np.round(z_scores[1:], 3),"p-value": np.round(p_values[1:], 3)})

results = results.sort_values(by="p-value");print("\nFinal Results:\n");print(results)

results.to_excel("Ridge_Final_Death_vs_Discharged.xlsx", index=False)

print("\nResults saved successfully.")
