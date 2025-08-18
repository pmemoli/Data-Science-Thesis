# Updates the samples from the benchmarks with internal metrics

from src.metrics.entropy import (
    predictive_entropy,
    shannon_entropy,
    attention_entropy,
)
import jsonlines
import torch
import os

benchmark_src_path = "src/data/evaluation_results"
tensor_src_path = "src/data/tensor_states"
metrics_src_path = "src/data/metric_results"

os.makedirs(metrics_src_path, exist_ok=True)

files = os.listdir(benchmark_src_path)
sample_files = [file for file in files if file.startswith(f"samples_")]

for file in sample_files:
    file_path = f"{benchmark_src_path}/{file}"

    results = []
    with jsonlines.open(file_path, mode="r") as reader:
        file_items = [item for item in reader]

    for item in file_items:
        item["file_name"] = file_path
        results.append(item)

    def load_tensor(unique_id: str):
        tensor_path = f"{tensor_src_path}/{unique_id}.pt"
        return torch.load(tensor_path, map_location="cpu")

    for result in results:
        unique_id = result.get("prompt_hash")
        if not unique_id:
            continue

        try:
            tensor_data = load_tensor(unique_id)
        except FileNotFoundError:
            print(f"Tensor data for {unique_id} not found, skipping.")
            continue

        print(f"Processing {unique_id}...")

        sequence_probabilities = tensor_data.get("sequence_probabilities")
        token_distribution = tensor_data.get("token_distribution")
        attentions = tensor_data.get("attentions")

        if (
            sequence_probabilities is None
            or token_distribution is None
            or attentions is None
        ):
            print(f"Incomplete tensor data for {unique_id}, skipping.")
            continue

        pe = predictive_entropy(sequence_probabilities)
        se = shannon_entropy(token_distribution)
        ae = attention_entropy(attentions)

        result["predictive_entropy"] = pe[0].item()
        result["shannon_entropy"] = se[0].item()
        result["attention_entropy"] = ae[0].item()

    # save results
    with jsonlines.open(file_path, mode="w") as writer:
        writer.write_all(results)

    print(f"Saved metrics to {file_path}")
