# %%
from src.metrics.sequence_ensemble import influence, attention_rollout
from sklearn.metrics import roc_auc_score
from typing import Literal
import numpy as np
import hashlib
import torch

tensor_data_filename = "src/data/runs/gsm-test/gsm8k_microsoft_Phi-3.5-mini-instruct_20250916-210355.pt"
tensor_data = torch.load(
    tensor_data_filename, map_location=torch.device("cpu"), mmap=True
)


def hash_result(question, response):
    hasher = hashlib.sha256()
    hasher.update(question.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(response.encode("utf-8"))
    return hasher.hexdigest()


# %% Descriptive statistics of the groups
def descriptive_statistics(entropy, influence):
    influence = influence / influence.sum()

    # Pure entropy based stats
    entropy_peak_threshold = 0.2
    high_entropy_mask = entropy > entropy_peak_threshold
    entropy_peaks = high_entropy_mask.sum().item()
    peak_density = entropy_peaks / entropy.shape[0]
    highest_entropy_peak = entropy.max().item()
    avg_entropy = entropy.mean().item()
    entropy_p75 = torch.quantile(entropy, 0.75).item()
    entropy_p90 = torch.quantile(entropy, 0.90).item()
    entropy_std = entropy.std().item()

    # Entropy mixed with influence stats
    influence_peak_threshold = 1 / entropy.shape[0]
    high_influence_mask = influence > influence_peak_threshold
    critical_mask = high_entropy_mask & high_influence_mask

    critical_peaks = critical_mask.sum().item()
    critical_peak_density = critical_peaks / entropy.shape[0]

    # Weighted entropy by attention
    weighted_entropy = entropy * influence
    avg_weighted_entropy = weighted_entropy.sum().item()

    weighted_entropy_std = weighted_entropy.std().item()
    highest_weighted_entropy_peak = weighted_entropy.max().item()
    max_weighted_entropy_jump = (
        torch.diff(weighted_entropy).abs().max().item()
        if weighted_entropy.shape[0] > 1
        else 0.0
    )

    return {
        # Basic entropy stats
        "avg_entropy": avg_entropy,
        "entropy_std": entropy_std,
        "entropy_p75": entropy_p75,
        "entropy_p90": entropy_p90,
        "entropy_peak_density": peak_density,
        "highest_entropy_peak": highest_entropy_peak,
        # Influence mixed stats
        "avg_weighted_entropy": avg_weighted_entropy,
        "weighted_entropy_std": weighted_entropy_std,
        "critical_peak_density": critical_peak_density,
        "highest_weighted_entropy_peak": highest_weighted_entropy_peak,
        "max_weighted_entropy_jump": max_weighted_entropy_jump,
    }


# %%
positive_agregate_stats = {}
negative_agregate_stats = {}

i = 0
for item in tensor_data:
    prompt_length = item["prompt_length"]

    stats = descriptive_statistics(item_entropy, item_influence)


# %%
# Compute average and standard deviation of the stats
avg_positive_stats = {
    k: np.mean(v) for k, v in positive_agregate_stats.items()
}
sd_positive_stats = {k: np.std(v) for k, v in positive_agregate_stats.items()}
avg_negative_stats = {
    k: np.mean(v) for k, v in negative_agregate_stats.items()
}
sd_negative_stats = {k: np.std(v) for k, v in negative_agregate_stats.items()}

print(f"{'='*60}")
print(f"{'HALLUCINATION DETECTION STATS':^60}")
print(f"{'='*60}")

for key in avg_positive_stats.keys():
    pos_mean, pos_sd = avg_positive_stats[key], sd_positive_stats[key]
    neg_mean, neg_sd = avg_negative_stats[key], sd_negative_stats[key]
    abs_diff = neg_mean - pos_mean
    cohen_denom = (pos_sd**2 + neg_sd**2) / 2
    cohen_d = abs_diff / np.sqrt(cohen_denom)

    if abs(neg_mean) > 1e-6:  # avoid division by zero
        rel_diff = (abs_diff / abs(neg_mean)) * 100
    else:
        rel_diff = float("inf") if abs_diff != 0 else 0

    y_true = np.concatenate(
        [
            np.zeros(len(positive_agregate_stats[key])),  # 0 for correct
            np.ones(
                len(negative_agregate_stats[key])
            ),  # 1 for incorrect (ERRORS - what we want to detect)
        ]
    )
    y_scores = np.concatenate(
        [positive_agregate_stats[key], negative_agregate_stats[key]]
    )

    # Highlight potentially discriminative features
    print(f"{key:<20}")
    print(f"   Correct:   {pos_mean:6.3f} ± {pos_sd:5.3f}")
    print(f"   Incorrect: {neg_mean:6.3f} ± {neg_sd:5.3f}")
    print(f"   Diff:      {abs_diff:+6.3f}")
    print(f"   RelDiff:   {rel_diff:6.2f}%")
    print(f"   Cohen's d: {cohen_d:6.3f}")
    print()
