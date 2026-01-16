
import numpy as np
import pandas as pd

data = np.load("xgb_test_inference.npz")

df = pd.DataFrame({
    "y_true": data["y"],
    "y_pred_proba": data["proba"],
    "y_pred_label": (data["proba"] >=  0.3564
).astype(int)
})

df.to_csv("test_predictions.csv", index=False)
from sklearn.metrics import roc_auc_score, f1_score

print("AUROC:", roc_auc_score(df.y_true, df.y_pred_proba))
print("F1:", f1_score(df.y_true, df.y_pred_label))
