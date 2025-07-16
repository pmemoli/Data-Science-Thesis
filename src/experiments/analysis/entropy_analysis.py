# %%
from sklearn import metrics
import pandas as pd

# %%
"""AUROC using entropy based metrics to detect failures on the gsm8k dataset"""
entropy_df = pd.read_csv(
    "src/experiments/results/gsm8k_train_evaluation_results.csv"
)

# %%
is_incorrect = entropy_df["is_incorrect"].astype(int).values

pe_scores = entropy_df["predictive_entropy"].values
pe_auroc = metrics.roc_auc_score(is_incorrect, pe_scores)

shannon_scores = entropy_df["shannon_entropy"].values
shannon_auroc = metrics.roc_auc_score(is_incorrect, shannon_scores)

# %%
print(f"Predictive Entropy AUROC: {pe_auroc:.4f}")
print(f"Shannon Entropy AUROC: {shannon_auroc:.4f}")
