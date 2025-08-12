# %%
%reload_ext autoreload
%autoreload 2

from src.metrics.entropy import (
    predictive_entropy,
    shannon_entropy,
    attention_entropy,
)
import jsonlines
import torch

# %% Analysis of GSM8K results
benchmark_src_path = "src/data/benchmark_results/"

with jsonlines.open(
    f"{benchmark_src_path}/gsm8k/samples_gsm8k_2025-08-11T20-51-25.059120.jsonl",
    mode="r",
) as reader:
    gsm8k_results = [item for item in reader]
    gsm8k_results = list(
        filter(
            lambda x: x["filter"] == "flexible-extract",
            gsm8k_results,
        )
    )


# %% Load tensors
def load_tensor(unique_id: str):
    tensor_path = f"{tensor_src_path}/{unique_id}.pt"
    return torch.load(tensor_path)


tensor_src_path = "src/data/tensor_states/"
unique_id = gsm8k_results[0]["prompt_hash"]

tensor_data = load_tensor(unique_id)
sample_data = gsm8k_results[0]

# %% Compute benchmark score
benchmark_score = sum(
    [int(result["exact_match"]) for result in gsm8k_results]
) / len(gsm8k_results)

print(benchmark_score)


# %% Compute output token distribution, sequence probabilities and attention matrix
sequence_probabilities = tensor_data["sequence_probabilities"]
token_distribution = tensor_data["token_distribution"]
attentions = tensor_data["attentions"]

pe = predictive_entropy(sequence_probabilities)
sa = shannon_entropy(token_distribution)
ae = attention_entropy(attentions)

ae, pe, sa
