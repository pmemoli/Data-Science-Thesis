# %%
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
import hashlib
import os


# %% Descriptive statistics of the groups
def descriptive_statistics(entropy, influence):
    # Pure entropy based stats
    entropy_peak_threshold = 0.2
    high_entropy_mask = entropy > entropy_peak_threshold
    entropy_peaks = high_entropy_mask.sum().item()
    peak_density = entropy_peaks / entropy.shape[0]
    avg_entropy = entropy.mean().item()

    # Entropy mixed with influence stats
    influence = influence / influence.sum()

    influence_peak_threshold = 1 / entropy.shape[0]
    high_influence_mask = influence > influence_peak_threshold
    critical_mask = high_entropy_mask & high_influence_mask

    critical_peaks = critical_mask.sum().item()
    critical_peak_density = critical_peaks / entropy.shape[0]

    # Weighted entropy by attention
    weighted_entropy = entropy * influence
    avg_weighted_entropy = weighted_entropy.sum().item()

    return {
        # Basic entropy stats
        "avg_entropy": avg_entropy,
        "entropy_peak_density": peak_density,
        # Influence mixed stats
        "avg_weighted_entropy": avg_weighted_entropy,
        "critical_peak_density": critical_peak_density,
    }


def hash_result(question):
    hasher = hashlib.sha256()
    hasher.update(question.encode("utf-8"))
    return hasher.hexdigest()


tensor_data_filename = "src/data/runs/gsm-exploration/gsm8k_microsoft_Phi-3.5-mini-instruct_20250916-210355.pt"
tensor_data = torch.load(
    tensor_data_filename, map_location=torch.device("cpu"), mmap=True
)

hashes = [hash_result(item["prompt"]) for item in tensor_data]

suite = "gsm-test"
tensor_path = f"src/data/runs/{suite}"
tensor_files = os.listdir(tensor_path)

positive_agregate_stats = {}
negative_agregate_stats = {}

i = 0
for file in tensor_files:
    full_path = f"{tensor_path}/{file}"
    tensor = torch.load(full_path)
    for tensor_item in tensor:
        # hash = hash_result(tensor_item["prompt"])
        # if hash in hashes:
        #     continue
        #
        print(f"Processing item {i+1}")

        entropy = tensor_item["token_shannon_entropy"].to(dtype=torch.float32)
        influence = tensor_item["attention_influence"]["rfn"]["rollout_09"].to(
            dtype=torch.float32
        )
        prompt_length = tensor_item["prompt_length"]

        stats = descriptive_statistics(
            entropy[prompt_length:], influence[prompt_length:]
        )

        if tensor_item["success"]:
            for key, value in stats.items():
                if key not in positive_agregate_stats:
                    positive_agregate_stats[key] = []

                positive_agregate_stats[key].append(value)

        else:
            for key, value in stats.items():
                if key not in negative_agregate_stats:
                    negative_agregate_stats[key] = []

                negative_agregate_stats[key].append(value)

        i += 1

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

    # Calculate AUROC
    # Higher feature values should indicate errors for good discrimination
    try:
        auroc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auroc = float("nan")  # In case of issues (e.g., all same values)

    # Highlight potentially discriminative features
    print(f"{key:<20}")
    print(f"   Correct:   {pos_mean:6.3f} ± {pos_sd:5.3f}")
    print(f"   Incorrect: {neg_mean:6.3f} ± {neg_sd:5.3f}")
    print(f"   Diff:      {abs_diff:+6.3f}")
    print(f"   RelDiff:   {rel_diff:6.2f}%")
    print(f"   Cohen's d: {cohen_d:6.3f}")
    print(f"   AUROC:     {auroc:6.3f}")
    print()
