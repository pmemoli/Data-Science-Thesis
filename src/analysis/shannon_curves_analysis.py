# %%
import matplotlib.pyplot as plt
import numpy as np
import hashlib
import torch

# %%
path = "src/analysis"
items = torch.load(
    f"{path}/shannon_entropy_gsm8k_microsoft_Phi-3.5-mini-instruct-1.pt"
)

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
def descriptive_statistics(data):
    entropy = data["shannon_entropy"][0, data["prompt_length"] :]
    peak_threshold = 0.1

    peaks = (entropy > peak_threshold).sum().item()

    avg_peak_height = (
        entropy[entropy > peak_threshold].mean().item() if peaks > 0 else 0.0
    )
    peak_density = peaks / entropy.shape[0]
    avg_entropy = entropy.mean().item()
    highest_peak = entropy.max().item()

    return {
        "num_peaks": peaks,
        "peak_density": peak_density,
        "avg_entropy": avg_entropy,
        "highest_peak": highest_peak,
        "avg_peak_height": avg_peak_height,
    }


# %%
positive_agregate_stats = {}
negative_agregate_stats = {}

for item in items:
    stats = descriptive_statistics(item)
    attention_stats = descriptive_statistics(item)

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

    # Store curves
    # plot_shannon_curve(item)


# Stats without attention weighting
avg_positive_stats = {
    k: sum(v) / len(v) for k, v in positive_agregate_stats.items()
}
sd_positive_stats = {
    k: float(np.std(v)) for k, v in positive_agregate_stats.items()
}

avg_negative_stats = {
    k: sum(v) / len(v) for k, v in negative_agregate_stats.items()
}
sd_negative_stats = {
    k: float(np.std(v)) for k, v in negative_agregate_stats.items()
}

print("\n")
print("Positive stats mean:")
print(avg_positive_stats, sep="\n")
# print("Positive stats std:")
# print(sd_positive_stats, sep="\n")

print("\n")
print("Negative stats:")
print(avg_negative_stats, sep="\n")
# print("Negative stats std:")
# print(sd_negative_stats, sep="\n")
