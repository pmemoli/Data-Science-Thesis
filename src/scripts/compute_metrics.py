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

datasets = ["gsm8k", "math"]

for dataset in datasets:
    print(f"Dataset: {dataset}")

    # Get last created file in the dataset directory
    files = os.listdir(benchmark_src_path)
    sample_files = [
        file for file in files if file.startswith(f"samples_{dataset}")
    ]
    if not sample_files:
        print(f"No sample files found for dataset {dataset}, skipping...")
        continue

    file = max(
        sample_files,
        key=lambda f: os.path.getctime(os.path.join(benchmark_src_path, f)),
    )

    with jsonlines.open(
        f"{benchmark_src_path}/{dataset}/{file}",
        mode="r",
    ) as reader:
        results = [item for item in reader]
        results = list(
            filter(
                lambda x: x["filter"] == "flexible-extract",
                results,
            )
        )

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
        tensor_data = load_tensor(unique_id)

        sequence_probabilities = tensor_data["sequence_probabilities"]
        token_distribution = tensor_data["token_distribution"]
        attentions = tensor_data["attentions"]

        incorrect_results.append(result["exact_match"] == 0)

        pe_values.append(predictive_entropy(sequence_probabilities).item())
        se_values.append(shannon_entropy(token_distribution).item())
        ae_values.append(attention_entropy(attentions).item())

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
