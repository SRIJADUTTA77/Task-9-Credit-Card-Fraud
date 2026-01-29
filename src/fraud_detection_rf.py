import pandas as pd

df = pd.read_csv("../dataset/creditcard.csv")

print(df.shape)
print(df.head())

print(df["Class"].value_counts())
from sklearn.metrics import roc_curve, auc, precision_recall_curve

X = df.drop("Class", axis=1)
y = df["Class"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=42
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("Logistic Regression Report")
print(classification_report(y_test, y_pred_lr))

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]
from sklearn.metrics import classification_report

print("Random Forest Report")
print(classification_report(y_test, y_pred_rf))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_rf)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix – Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("../outputs/confusion_matrix.png")
plt.close()

import numpy as np

importances = rf.feature_importances_
indices = np.argsort(importances)[-10:]

plt.figure(figsize=(8,6))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), X.columns[indices])
plt.title("Top Fraud Indicators")
plt.savefig("../outputs/feature_importance.png")
plt.close()

import joblib

joblib.dump(rf, "../models/random_forest_fraud.pkl")
# =========================
# ROC CURVE
# =========================
fpr, tpr, _ = roc_curve(y_test, y_proba_rf)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()

plt.savefig("../outputs/roc_curve.png", bbox_inches="tight")
plt.close()

# =========================
# PRECISION–RECALL CURVE
# =========================
precision, recall, _ = precision_recall_curve(y_test, y_proba_rf)

plt.figure(figsize=(7, 6))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve - Random Forest")

plt.savefig("../outputs/precision_recall_curve.png", bbox_inches="tight")
plt.close()
