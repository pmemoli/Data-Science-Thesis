from src.metrics.entropy import (
    predictive_entropy,
    shannon_entropy,
    attention_entropy,
)
from sklearn.metrics import roc_auc_score
import json
import time
import jsonlines
import torch
import os

benchmark_src_path = "src/data/evaluation_results"
tensor_src_path = "src/data/tensor_states"
metrics_src_path = "src/data/metric_results"

datasets = ["gsm8k", "hendrycks_math"]

for dataset in datasets:
    print(f"Dataset: {dataset}")

    # Get last created file in the dataset directory
    files = os.listdir(benchmark_src_path)
    sample_files = [
        file for file in files if file.startswith(f"samples_{dataset}")
    ]

    if not sample_files:
        continue

    results = []

    for file in sample_files:
        with jsonlines.open(
            f"{benchmark_src_path}/{file}",
            mode="r",
        ) as reader:
            file_items = [item for item in reader]

            if dataset == "gsm8k":
                file_items = list(
                    filter(
                        lambda x: x["filter"] == "flexible-extract",
                        file_items,
                    )
                )
                results.extend(file_items)

            elif dataset == "hendrycks_math":
                # The exact match simply doesn't work, some manual parsing should be done
                results.extend(file_items)

    # Benchmark score
    benchmark_score = sum(
        [int(result["exact_match"]) for result in results]
    ) / len(results)

    # Compute and store metrics
    metrics = {
        "dataset": dataset,
        "benchmark_score": benchmark_score,
        "se": 0.0,
        "se_auroc": 0.0,
        "pe": 0.0,
        "pe_auroc": 0.0,
        "ae": 0.0,
        "ae_auroc": 0.0,
    }

    def load_tensor(unique_id: str):
        tensor_path = f"{tensor_src_path}/{unique_id}.pt"
        return torch.load(tensor_path, map_location="cpu")

    pe_values = []
    se_values = []
    ae_values = []
    incorrect_results = []
    for result in results:
        unique_id = result["prompt_hash"]

        try:
            tensor_data = load_tensor(unique_id)
        except:
            print(f"Tensor data for {unique_id} not found, skipping.")
            continue

        print(f"Processing {unique_id}...")

        sequence_probabilities = tensor_data["sequence_probabilities"]
        token_distribution = tensor_data["token_distribution"]
        attentions = tensor_data["attentions"]

        incorrect_results.append(result["exact_match"] == 0)

        pe = predictive_entropy(sequence_probabilities)
        se = shannon_entropy(token_distribution)
        ae = attention_entropy(attentions)

        pe_values.append(pe[0].item())
        se_values.append(se[0].item())
        ae_values.append(ae[0].item())

    metrics["pe"] = sum(pe_values) / len(pe_values)
    metrics["se"] = sum(se_values) / len(se_values)
    metrics["ae"] = sum(ae_values) / len(ae_values)

    metrics["pe_auroc"] = roc_auc_score(incorrect_results, pe_values)
    metrics["se_auroc"] = roc_auc_score(incorrect_results, se_values)
    metrics["ae_auroc"] = roc_auc_score(incorrect_results, ae_values)

    # store json
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    metrics_file = f"{metrics_src_path}/{dataset}_metrics_{timestamp}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
