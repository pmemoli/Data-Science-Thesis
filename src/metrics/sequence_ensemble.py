import torch
from typing import Literal
import gc

eps = 1e-8

# Token-pooling function
WeightMethod = Literal[
    "entropy",
    "prob",
    "attention_rollout_mean",
    "attention_rollout_last",
    "attention_influence_mean",
    "attention_influence_last",
]


def attention_rollout(
    attentions: torch.Tensor,
    residual_stream_proportion: float = 0.5,
    attention_output_proportion: float = 0.5,
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

    # Aggregate heads to [num_layers, batch_size, total_length, total_length]
    attentions = attentions.mean(dim=2)  # type: ignore

    # Normalize attention to take into account residual connections
    sequence_length = attentions.size(-1)
    identity = torch.eye(sequence_length).to(attentions.device)
    attentions = (
        attention_output_proportion * attentions
        + residual_stream_proportion * identity.unsqueeze(0).unsqueeze(0)
    )

    # Recursively multiply the weight matrices
    rollout = attentions[0, :, :, :]
    num_layers = attentions.size(0)
    for i in range(1, num_layers):
        rollout = torch.bmm(attentions[i, :, :, :], rollout)

    return rollout  # type: ignore


def influence(
    attentions: torch.Tensor,
    hidden_states: torch.Tensor,
    attention_outputs: torch.Tensor,
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

    # Aggregate heads to [num_layers, batch_size, total_length, total_length]
    attentions = attentions.mean(dim=2)  # type: ignore

    # Normalize the attention matrix to take into account the residual connections
    # [num_layers, batch_size, total_length]
    hidden_state_norms = torch.linalg.norm(hidden_states, dim=-1)[:-1]
    attention_output_norms = torch.linalg.norm(attention_outputs, dim=-1)

    # [num_layers, batch_size, total_length, total_length]
    hs_norm_matrix = torch.diag_embed(hidden_state_norms)
    ao_norm_matrix = torch.diag_embed(attention_output_norms)
    normalization_matrix = torch.diag_embed(
        1 / (attention_output_norms + hidden_state_norms + eps)
    )

    num_layers = attentions.size(0)
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

    # Influence algorithm
    influence = attentions[0, :, :, :]
    for i in range(1, num_layers):
        influence = torch.bmm(attentions[i, :, :, :], influence)

    return influence


def sequence_ensemble(
    metric: torch.Tensor,  # [batch_size, sequence_length]
    last_layer_distribution: torch.Tensor,
    attentions: torch.Tensor,
    sequences: torch.Tensor,
    pad_token_id: int,
    pooling_ratio: float = 1,
    prompt_length: int = 0,
    weighting: None | WeightMethod = None,
):
    """
    Pool the uncertainty metric over the generated tokens based on different criterions.

    -metric: [batch_size, sequence_length]
    -last_layer_distribution: [batch_size, sequence_length, vocab_size]
    -output_mask: None or [batch_size, sequence_length] with 1 for valid tokens and 0 for padding/eos
    -pooling_ratio: float between 0 and 1 indicating the ratio of tokens to pool
    -weighting: None or "entropy" or "prob" to weight the metric by the token

    returns [batch_size]
    """

    output_mask = (sequences != pad_token_id).float()

    with torch.no_grad():
        if not weighting:
            pass

        elif weighting == "entropy":
            weights = -torch.sum(
                last_layer_distribution
                * torch.log(last_layer_distribution + 1e-8),
                dim=-1,
            )

        elif weighting == "prob":
            max_probs, _ = torch.max(last_layer_distribution, dim=-1)
            weights = 1 - max_probs

        elif "attention_rollout" in weighting:
            rollout = attention_rollout(attentions)
            if "mean" in weighting:
                weights = torch.mean(rollout[:, :, prompt_length:], dim=-1)

            elif "last" in weighting:
                weights = rollout[:, -1, prompt_length:]

        pool_amount = max(1, int(pooling_ratio * metric.shape[1]))
        top_k_values, top_k_ids = torch.topk(metric, pool_amount, dim=-1)

        selected_mask_values = torch.gather(
            output_mask, dim=-1, index=top_k_ids
        )
        top_k_values = torch.where(
            selected_mask_values == 0, torch.nan, top_k_values
        )

        # Weighted average
        if weighting:
            selected_weights = torch.gather(weights, dim=-1, index=top_k_ids)
            selected_weights = torch.where(
                selected_mask_values == 0, torch.nan, selected_weights
            )

            result = torch.nansum(
                top_k_values * selected_weights, dim=-1
            ) / torch.nansum(selected_weights, dim=-1)

        else:
            result = torch.nanmean(top_k_values, dim=-1)

    torch.cuda.empty_cache()
    gc.collect()

    return result
