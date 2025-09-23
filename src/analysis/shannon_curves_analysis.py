# %%
from src.metrics.sequence_ensemble import influence
import matplotlib.pyplot as plt
import numpy as np
import hashlib
import torch

# %%
path = "src/analysis"
items = torch.load(f"{path}/tensors.pt")

# %%
tensor_data_filename = "src/data/runs/validation/gsm8k_microsoft_Phi-3.5-mini-instruct_20250916-210355.pt"
tensor_data = torch.load(
    tensor_data_filename, map_location=torch.device("cpu"), mmap=True
)


# %%
def hash_result(question, response):
    hasher = hashlib.sha256()
    hasher.update(question.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(response.encode("utf-8"))
    return hasher.hexdigest()


# %% Stores the curves
def plot_shannon_curve(data):
    is_correct = data["is_correct"]
    prompt_length = data["prompt_length"]
    shannon_entropies = data["shannon_entropy"][0, prompt_length:]

    # Compute store path
    if is_correct:
        store_path = f"{path}/positive_curves/{hash_result(data['prompt'], data['generation'])}.png"
    else:
        store_path = f"{path}/negative_curves/{hash_result(data['prompt'], data['generation'])}.png"

    # Create the figure and plot
    plt.figure()
    plt.plot(shannon_entropies)
    plt.xlabel("Token Index (after prompt)")
    plt.ylabel("Shannon Entropy")
    plt.title("Shannon Entropy Curve")

    # Save and close
    plt.savefig(store_path, bbox_inches="tight")
    plt.close()


# %% Descriptive statistics of the groups
def descriptive_statistics(entropy, influence):
    influence = influence / influence.sum()

    # Pure entropy based stats
    entropy_peak_threshold = 0.5
    high_entropy_mask = entropy > entropy_peak_threshold
    entropy_peaks = high_entropy_mask.sum().item()
    avg_entropy_peak_height = (
        entropy[high_entropy_mask].mean().item() if entropy_peaks > 0 else 0.0
    )
    peak_density = entropy_peaks / entropy.shape[0]
    highest_entropy_peak = entropy.max().item()
    avg_entropy = entropy.mean().item()
    entropy_p75 = torch.quantile(entropy, 0.75).item()

    # Additional entropy-only statistics
    entropy_std = entropy.std().item()
    entropy_median = torch.median(entropy).item()

    # Additional percentiles
    entropy_p90 = torch.quantile(entropy, 0.90).item()

    # Entropy gradient statistics (local changes)
    if entropy.shape[0] > 1:
        entropy_diffs = torch.diff(entropy)
        max_entropy_jump = entropy_diffs.abs().max().item()
    else:
        max_entropy_jump = 0.0

    # Entropy mixed with influence stats
    influence_peak_threshold = 1 / entropy.shape[0]
    high_influence_mask = influence > influence_peak_threshold
    critical_mask = high_entropy_mask & high_influence_mask
    critical_peaks = critical_mask.sum().item()
    critical_peak_density = critical_peaks / entropy.shape[0]
    avg_critical_peak_height = (
        entropy[critical_mask].mean().item() if critical_peaks > 0 else 0.0
    )
    highest_critical_peak = (
        entropy[critical_mask].max().item() if critical_peaks > 0 else 0.0
    )

    # Weighted entropy by attention
    weighted_entropy = entropy * influence
    avg_weighted_entropy = weighted_entropy.sum().item()

    weighted_entropy_p75 = torch.quantile(weighted_entropy, 0.75).item()
    weighted_entropy_p90 = torch.quantile(weighted_entropy, 0.90).item()

    return {
        # Basic entropy stats
        "avg_entropy": avg_entropy,
        "entropy_std": entropy_std,
        "entropy_median": entropy_median,
        # Percentiles
        "entropy_p75": entropy_p75,
        "entropy_p90": entropy_p90,
        # Peak analysis
        "num_entropy_peaks": entropy_peaks,
        "entropy_peak_density": peak_density,
        "highest_entropy_peak": highest_entropy_peak,
        "avg_entropy_peak_height": avg_entropy_peak_height,
        # Change/smoothness metrics
        "max_entropy_jump": max_entropy_jump,
        # Mixed stats (with influence)
        "avg_weighted_entropy": avg_weighted_entropy,
        "num_critical_peaks": critical_peaks,
        "critical_peak_density": critical_peak_density,
        "avg_critical_peak_height": avg_critical_peak_height,
        "highest_critical_peak": highest_critical_peak,
        # Weighted percentiles
        "weighted_entropy_p75": weighted_entropy_p75,
        "weighted_entropy_p90": weighted_entropy_p90,
    }


# %%
positive_agregate_stats = {}
negative_agregate_stats = {}

i = 0
for item in items:
    hash_id = hash_result(item["prompt"], item["generation"])
    print(f"Processing item {i+1}/{len(items)}")

    item_tensor_data = next(
        x
        for x in tensor_data
        if hash_result(x["prompt"], x["generation"]) == hash_id
    )

    attentions = item_tensor_data["attentions"]
    hidden_states = item_tensor_data["hidden_states"]
    attention_outputs = item_tensor_data["attention_outputs"]
    prompt_length = item_tensor_data["prompt_length"]

    item_influence = influence(
        attentions,
        hidden_states,
        attention_outputs,
        difference="angle",
        receptive_field_norm=True,
    )[0].mean(dim=0)[prompt_length:]
    item_entropy = item["shannon_entropy"][0, prompt_length:]

    stats = descriptive_statistics(item_entropy, item_influence)

    # Aggregate stats
    if item["is_correct"]:
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

    # Store curves
    # plot_shannon_curve(item)

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

    # Highlight potentially discriminative features
    print(f"{key:<20}")
    print(f"   Correct: {pos_mean:6.3f} ± {pos_sd:5.3f}")
    print(f"   Incorrect: {neg_mean:6.3f} ± {neg_sd:5.3f}")
    print(f"   Diff:     {abs_diff:+6.3f}")
    print(f"   RelDiff:  {rel_diff:6.2f}%")
    print(f"   Cohen's d: {cohen_d:6.3f}")

    print()
