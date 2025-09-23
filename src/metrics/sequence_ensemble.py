import gc
from typing import Literal
import torch

eps = 1e-8

weighting_options = [
    "prob",
    "attention_rollout",
    "attention_influence_norm",
    "attention_influence_angle",
    "attention_influence_projection",
]


eps = 1e-8


def attention_rollout(
    attentions: torch.Tensor,
    residual_stream_proportion: float = 0.5,
    attention_output_proportion: float = 0.5,
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

    # Aggregate heads to [num_layers, batch_size, total_length, total_length]
    attentions = attentions.mean(dim=2)  # type: ignore

    # Weight by the amount of tokens that can attend to it
    if receptive_field_norm:
        n = attentions.size(-1)
        norm_matrix = torch.zeros(n, n)
        i_indices = torch.arange(n).unsqueeze(1).expand(n, n)
        j_indices = torch.arange(n).unsqueeze(0).expand(n, n)

        norm_matrix.fill_diagonal_(1)

        lower_mask = i_indices > j_indices
        norm_matrix[lower_mask] = 1.0 / (
            i_indices[lower_mask] - j_indices[lower_mask] + 1
        )

        norm_matrix = norm_matrix[None, None, :, :].to(attentions.device)
        attentions = attentions * norm_matrix

        # Normalize again
        attentions = attentions / attentions.sum(dim=-1, keepdim=True)

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

    # Aggregate heads to [num_layers, batch_size, total_length, total_length]
    attentions = attentions.mean(dim=2)  # type: ignore

    # Weight by the amount of tokens that can attend to it
    if receptive_field_norm:
        n = attentions.size(-1)
        norm_matrix = torch.zeros(n, n)
        i_indices = torch.arange(n).unsqueeze(1).expand(n, n)
        j_indices = torch.arange(n).unsqueeze(0).expand(n, n)

        norm_matrix.fill_diagonal_(1)

        lower_mask = i_indices > j_indices
        norm_matrix[lower_mask] = 1.0 / (
            i_indices[lower_mask] - j_indices[lower_mask] + 1
        )

        norm_matrix = norm_matrix[None, None, :, :].to(attentions.device)
        attentions = attentions * norm_matrix

        # Normalize again
        attentions = attentions / attentions.sum(dim=-1, keepdim=True)

    # Normalize the attention matrix to take into account the residual connections
    num_layers = attentions.size(0)
    hidden_states = hidden_states[:-1]  # removes output hidden states

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


def sequence_ensemble(
    metric: torch.Tensor,  # [batch_size, sequence_length]
    last_layer_distribution: torch.Tensor,
    attentions: torch.Tensor,
    attention_outputs: torch.Tensor,
    hidden_states: torch.Tensor,
    pooling_ratio: float = 1,
    prompt_length: int = 0,
    weighting: None | str = None,
    quantile: float | None = None,
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

    with torch.no_grad():
        metric = metric[:, prompt_length:]

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
            rollout = attention_rollout(attentions, 0.96, 0.04)
            weights = torch.mean(rollout, dim=1)

        elif "attention_influence" in weighting:
            if "norm" in weighting:
                mode = "norm"
            elif "angle" in weighting:
                mode = "angle"
            else:
                mode = "projection"

            infl = influence(
                attentions, hidden_states, attention_outputs, mode
            )
            weights = torch.mean(infl, dim=1)

        pool_amount = max(1, int(pooling_ratio * metric.shape[1]))

        # Weight the metric
        if weighting:
            weights = weights[:, prompt_length:]

            # If weighting, choose the top k according to the weights
            top_k_weights, top_k_ids = torch.topk(weights, pool_amount, dim=-1)
            top_k_values = torch.gather(metric, dim=-1, index=top_k_ids)

        else:
            top_k_values, top_k_ids = torch.topk(metric, pool_amount, dim=-1)

        if quantile:
            result = torch.quantile(top_k_values, quantile, dim=-1)

        else:
            if weighting:
                top_k_values = (
                    top_k_values
                    * top_k_weights
                    / torch.sum(top_k_weights, dim=-1)
                )

            else:
                top_k_values = top_k_values / pool_amount

            result = torch.sum(top_k_values, dim=-1)

    torch.cuda.empty_cache()
    gc.collect()

    return result
