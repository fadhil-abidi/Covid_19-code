# Ridge vs Neural Network Comparison when Death = 1 , Discharged = 0
#1- import libraris and functions
import pandas as pd;import numpy as np;import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier;from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve;from scipy import stats

# 1- read data from Excel file you can change to your data file ️ 
file_path = "COVID19_Patient_Sample_Imputed_200.xlsx";df = pd.read_excel(file_path)

# Convert outcome if needed
if df["outcome"].dtype == "object":
    df["outcome"] = df["outcome"].str.lower().map({"death":1,"discharged":0})

# Convert categorical predictors
df = pd.get_dummies(df, drop_first=True);y = df["outcome"]
X = df.drop(columns=["outcome"])

# 2- Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 3- Standardization
scaler = StandardScaler();X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4- Ridge Model

ridge = LogisticRegressionCV(penalty="l2",solver="liblinear",cv=5,
    scoring="roc_auc",max_iter=3000)
ridge.fit(X_train, y_train)
ridge_prob = ridge.predict_proba(X_test)[:,1];ridge_auc = roc_auc_score(y_test, ridge_prob)

# 5- Neural Network Model
nn = MLPClassifier(hidden_layer_sizes=(20,10),activation='relu',
    max_iter=2000,random_state=42)
nn.fit(X_train, y_train)
nn_prob = nn.predict_proba(X_test)[:,1];nn_auc = roc_auc_score(y_test, nn_prob)

print("\nRidge AUC:", round(ridge_auc,3))
print("\nNeural Network AUC:", round(nn_auc,3))

# 6- DeLong Test (Approximation using correlated AUCs)

def delong_test(y_true, pred1, pred2):
    auc1 = roc_auc_score(y_true, pred1)
    auc2 = roc_auc_score(y_true, pred2)
    n = len(y_true)
    var = (auc1*(1-auc1) + auc2*(1-auc2)) / n
    z = (auc1 - auc2) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p

z_stat, p_value = delong_test(y_test, ridge_prob, nn_prob)

print("\nDeLong Test Z:", round(z_stat,3))
print("DeLong Test p-value:", round(p_value,4))

# 7- ROC Curves Plot
fpr_r, tpr_r, _ = roc_curve(y_test, ridge_prob);fpr_n, tpr_n, _ = roc_curve(y_test, nn_prob)

plt.figure(figsize=(7,6))
plt.plot(fpr_r, tpr_r, label=f"Ridge (AUC={ridge_auc:.3f})")
plt.plot(fpr_n, tpr_n, label=f"Neural Network (AUC={nn_auc:.3f})")
plt.plot([0,1],[0,1])
plt.xlabel("False Positive Rate");plt.ylabel("True Positive Rate");plt.title("ROC Curve Comparison")
plt.legend();plt.show()
