# %%
from src.metrics.sequence_ensemble import influence
import matplotlib.pyplot as plt
import numpy as np
import hashlib
import torch

path = "src/analysis"
items = torch.load(f"{path}/tensors.pt")

tensor_data_filename = "src/data/runs/validation/gsm8k_microsoft_Phi-3.5-mini-instruct_20250916-210355.pt"
tensor_data = torch.load(
    tensor_data_filename, map_location=torch.device("cpu"), mmap=True
)


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


# %%
i = 0
for item in items:
    # Store curves
    plot_shannon_curve(item)
