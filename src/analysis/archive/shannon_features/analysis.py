# %%

# %%
from src.metrics.sequence_ensemble import influence
from sklearn.metrics import roc_auc_score
from typing import Literal
import numpy as np
import hashlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=False,
)

eps = 1e-8

path = "src/analysis/shannon_features"
items = torch.load(f"{path}/tensors.pt")

tensor_data_filename = "src/data/runs/gsm-exploration/gsm8k_microsoft_Phi-3.5-mini-instruct_20250916-210355.pt"
tensor_data = torch.load(
    tensor_data_filename, map_location=torch.device("cpu"), mmap=True
)


def hash_result(question, response):
    hasher = hashlib.sha256()
    hasher.update(question.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(response.encode("utf-8"))
    return hasher.hexdigest()


# %%
import torch


def compute_macs(attentions, epsilon=0.87):
    """
    Compute Multi-Layer Attention Consistency Scores (MACS).

    Args:
        attentions: Shape [num_layers, batch, heads, seq_len, seq_len]
        input_length: Number of input tokens (N)
        epsilon: Floor vector coefficient (default: 0.05)

    Returns:
        z_scores: Token importance Z-scores, shape [batch, input_length]
        raw_consistency: Raw consistency scores, shape [batch, input_length]
    """
    if isinstance(attentions, list):
        attentions = torch.stack(attentions)

    num_layers = attentions.size(0)

    consistency = None
    for layer_idx in range(num_layers):
        # Max-pool across heads [batch, seq_len, seq_len]
        m_prime = attentions[layer_idx].max(dim=1).values

        # Apply floor vector
        m = (1 - epsilon) * m_prime + epsilon
        # m = 1 + m_prime

        # Multiplicative aggregation
        consistency = m if consistency is None else consistency * m

    raw_consistency = consistency.pow(1 / num_layers)

    # Normalize to Z-scores
    mean = raw_consistency.mean(dim=-1, keepdim=True)
    std = raw_consistency.std(dim=-1, keepdim=True)
    z_scores = (raw_consistency - mean) / (std + 1e-8)
    z_scores = z_scores - z_scores.min(dim=-1, keepdim=True).values + 1e-8

    return z_scores, raw_consistency


def influence(
    attentions: torch.Tensor,
    hidden_states: torch.Tensor,
    attention_outputs: torch.Tensor,
    difference: Literal["norm", "angle", "projection"] = "norm",
    receptive_field_norm: bool = False,
) -> torch.Tensor:
    """
    Perform attention rollout as described in the paper:
    https://aclanthology.org/2020.acl-main.385.pdf

    Input:
    attentions (torch.Tensor): Tensor of shape
    [num_layers, batch_size, num_heads, total_length, total_length]

    Output:
    torch.Tensor: Tensor of shape
    [batch_size, total_length, total_length]
    """

    dtype = attentions.dtype

    # Convert all inputs to specified dtype
    attentions = attentions.to(dtype=dtype)
    hidden_states = hidden_states.to(dtype=dtype)
    attention_outputs = attention_outputs.to(dtype=dtype)

    # Aggregate heads to [num_layers, batch_size, total_length, total_length]
    attentions = attentions.max(dim=2).values  # type: ignore

    # Weight by the amount of tokens that can attend to it
    # Normalize the attention matrix to take into account the residual connections
    num_layers = attentions.size(0)
    hidden_states = hidden_states[:-1]  # removes output hidden states

    if receptive_field_norm:
        n = attentions.size(-1)

        # Receptive field size for each column: [n, n-1, n-2, ..., 1]
        receptive_field = torch.arange(
            n, 0, -1, dtype=dtype, device=attentions.device
        )

        # Divide each column by its receptive field size
        attentions = attentions / receptive_field[None, None, None, :]

        # Renormalize rows
        attentions = attentions / attentions.sum(dim=-1, keepdim=True)

    # if receptive_field_norm:
    #     n = attentions.size(-1)
    #     norm_matrix = torch.zeros(n, n, dtype=dtype, device=attentions.device)
    #     i_indices = (
    #         torch.arange(n, dtype=dtype, device=attentions.device)
    #         .unsqueeze(1)
    #         .expand(n, n)
    #     )
    #     j_indices = (
    #         torch.arange(n, dtype=dtype, device=attentions.device)
    #         .unsqueeze(0)
    #         .expand(n, n)
    #     )
    #
    #     norm_matrix.fill_diagonal_(1)
    #
    #     lower_mask = i_indices > j_indices
    #     norm_matrix[lower_mask] = 1.0 / (
    #         i_indices[lower_mask] - j_indices[lower_mask] + 1
    #     )
    #
    #     norm_matrix = norm_matrix[None, None, :, :]
    #     attentions = attentions * norm_matrix
    #
    #     # Normalize again
    #     attentions = attentions / attentions.sum(dim=-1, keepdim=True)

    if difference == "norm":
        # [num_layers, batch_size, total_length]
        hidden_state_norms = torch.linalg.norm(hidden_states, dim=-1)
        attention_output_norms = torch.linalg.norm(attention_outputs, dim=-1)

        # [num_layers, batch_size, total_length, total_length]
        hs_norm_matrix = torch.diag_embed(hidden_state_norms)
        ao_norm_matrix = torch.diag_embed(attention_output_norms)
        normalization_matrix = torch.diag_embed(
            1 / (attention_output_norms + hidden_state_norms + eps)
        )

        for layer_idx in range(num_layers):
            attentions[layer_idx] = torch.bmm(
                ao_norm_matrix[layer_idx], attentions[layer_idx]
            )
            attentions[layer_idx] = (
                hs_norm_matrix[layer_idx] + attentions[layer_idx]
            )
            attentions[layer_idx] = torch.bmm(
                normalization_matrix[layer_idx], attentions[layer_idx]
            )

    elif difference == "angle":
        sum = hidden_states + attention_outputs
        hs_angle = torch.cosine_similarity(hidden_states, sum, dim=-1)
        ao_angle = torch.cosine_similarity(attention_outputs, sum, dim=-1)

        hs_angle_matrix = torch.diag_embed(hs_angle)
        ao_angle_matrix = torch.diag_embed(ao_angle)
        normalization_matrix = torch.diag_embed(
            1 / (hs_angle + ao_angle + eps)
        )

        for layer_idx in range(num_layers):
            attentions[layer_idx] = torch.bmm(
                ao_angle_matrix[layer_idx], attentions[layer_idx]
            )
            attentions[layer_idx] = (
                hs_angle_matrix[layer_idx] + attentions[layer_idx]
            )
            attentions[layer_idx] = torch.bmm(
                normalization_matrix[layer_idx], attentions[layer_idx]
            )

    elif difference == "projection":

        def projection(a, b):
            return torch.sum(a * b, dim=-1) / (
                torch.linalg.norm(b, dim=-1) ** 2
            )

        sum = hidden_states + attention_outputs
        hs_proj = projection(hidden_states, sum)
        ao_proj = projection(attention_outputs, sum)

        hs_proj_matrix = torch.diag_embed(hs_proj)
        ao_proj_matrix = torch.diag_embed(ao_proj)
        normalization_matrix = torch.diag_embed(1 / (hs_proj + ao_proj + eps))

        for layer_idx in range(num_layers):
            attentions[layer_idx] = torch.bmm(
                ao_proj_matrix[layer_idx], attentions[layer_idx]
            )
            attentions[layer_idx] = (
                hs_proj_matrix[layer_idx] + attentions[layer_idx]
            )
            attentions[layer_idx] = torch.bmm(
                normalization_matrix[layer_idx], attentions[layer_idx]
            )

    # Influence algorithm
    influence = attentions[0, :, :, :]
    for i in range(1, num_layers):
        influence = torch.bmm(attentions[i, :, :, :], influence)

    return influence


def aggregate_attention_influence(influence: torch.Tensor) -> torch.Tensor:
    n = influence.size(-1)
    aggregated_influence = influence.sum(dim=0)
    divisors = torch.arange(
        n, 0, -1, dtype=influence.dtype, device=influence.device
    )
    aggregated_influence = aggregated_influence / divisors

    return aggregated_influence


# %% Descriptive statistics of the groups
def descriptive_statistics(entropy, influence):
    influence = influence / influence.sum()

    # Pure entropy based stats
    entropy_peak_threshold = 0.3
    high_entropy_mask = entropy > entropy_peak_threshold
    entropy_peaks = high_entropy_mask.sum().item()
    peak_density = entropy_peaks / entropy.shape[0]
    avg_entropy = entropy.mean().item()

    # Entropy mixed with influence stats
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
        "peak_density": peak_density,
        # Influence mixed stats
        "avg_weighted_entropy": avg_weighted_entropy,
        "critical_peak_density": critical_peak_density,
    }


# %%
import string


def clean_influence(
    influence: torch.Tensor, token_sequence: list[int], tokenizer
):
    # influence: [seq_len]
    special_tokens = {"<0x0A>", "<|end|>"}
    punctuation = set(string.punctuation) | {"，", "。", "、", "！", "？"}

    tokens = [
        tokenizer.convert_ids_to_tokens(token_id)
        for token_id in token_sequence
    ]

    for i, token in enumerate(tokens):
        if (
            token in special_tokens
            or token.replace("▁", " ").strip() in punctuation
        ):
            influence[i] = 0

    return influence


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

    # item_influence = influence(
    #     attentions,
    #     hidden_states,
    #     attention_outputs,
    #     difference="projection",
    #     receptive_field_norm=True,
    # )[0][prompt_length:]

    # item_influence = attention_rollout(
    #     attentions,
    #     attention_output_proportion=0.1,
    #     residual_stream_proportion=0.9,
    #     receptive_field_norm=True,
    # )[0]

    zscores, consistency = compute_macs(attentions, epsilon=0.02)
    # item_influence = zscores[0].max(dim=0).values[prompt_length:]
    # item_influence = aggregate_attention_influence(consistency[0])[
    #     prompt_length:
    # ]
    item_influence = zscores[0].max(dim=0).values[prompt_length:]

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
