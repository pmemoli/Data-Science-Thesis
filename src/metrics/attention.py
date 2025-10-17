from typing import Literal
import torch

eps = 1e-8


def attention_rollout(
    attentions: torch.Tensor,
    residual_stream_proportion: float = 0.5,
    attention_output_proportion: float = 0.5,
    receptive_field_norm: None | Literal["column", "element"] = None,
    pool: Literal["max", "mean"] = "mean",
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

    # Convert to specified dtype
    attentions = attentions.to(dtype=dtype)

    # Aggregate heads to [num_layers, batch_size, total_length, total_length]
    if pool == "max":
        attentions, _ = attentions.max(dim=2)  # type: ignore
    else:
        attentions = attentions.mean(dim=2)  # type: ignore

    # Weight by the amount of tokens that can attend to it
    if receptive_field_norm == "element":
        n = attentions.size(-1)
        norm_matrix = torch.zeros(n, n, dtype=dtype, device=attentions.device)
        i_indices = (
            torch.arange(n, dtype=dtype, device=attentions.device)
            .unsqueeze(1)
            .expand(n, n)
        )
        j_indices = (
            torch.arange(n, dtype=dtype, device=attentions.device)
            .unsqueeze(0)
            .expand(n, n)
        )

        norm_matrix.fill_diagonal_(1)

        lower_mask = i_indices > j_indices
        norm_matrix[lower_mask] = 1.0 / (
            i_indices[lower_mask] - j_indices[lower_mask] + 1
        )

        norm_matrix = norm_matrix[None, None, :, :]
        attentions = attentions * norm_matrix

        # Normalize again
        attentions = attentions / attentions.sum(dim=-1, keepdim=True)

    elif receptive_field_norm == "column":
        n = attentions.size(-1)

        # Receptive field size for each column: [n, n-1, n-2, ..., 1]
        receptive_field = torch.arange(
            n, 0, -1, dtype=dtype, device=attentions.device
        )

        # Divide each column by its receptive field size
        attentions = attentions / receptive_field[None, None, None, :]

        # Renormalize rows
        attentions = attentions / attentions.sum(dim=-1, keepdim=True)

    # Normalize attention to take into account residual connections
    sequence_length = attentions.size(-1)
    identity = torch.eye(
        sequence_length, dtype=dtype, device=attentions.device
    )
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
    proportion: Literal["norm", "projection"] = "norm",
    receptive_field_norm: None | Literal["column", "element"] = None,
    pool: Literal["max", "mean"] = "mean",
) -> torch.Tensor:
    """
    Modify the rollout algorithm so that weighting is done dynamically, based on the porportion of attention output over the residual + attention_output sum

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
    if pool == "max":
        attentions, _ = attentions.max(dim=2)  # type: ignore
    else:
        attentions = attentions.mean(dim=2)  # type: ignore

    # Weight by the amount of tokens that can attend to it
    if receptive_field_norm == "element":
        n = attentions.size(-1)
        norm_matrix = torch.zeros(n, n, dtype=dtype, device=attentions.device)
        i_indices = (
            torch.arange(n, dtype=dtype, device=attentions.device)
            .unsqueeze(1)
            .expand(n, n)
        )
        j_indices = (
            torch.arange(n, dtype=dtype, device=attentions.device)
            .unsqueeze(0)
            .expand(n, n)
        )

        norm_matrix.fill_diagonal_(1)

        lower_mask = i_indices > j_indices
        norm_matrix[lower_mask] = 1.0 / (
            i_indices[lower_mask] - j_indices[lower_mask] + 1
        )

        norm_matrix = norm_matrix[None, None, :, :]
        attentions = attentions * norm_matrix

        # Normalize again
        attentions = attentions / attentions.sum(dim=-1, keepdim=True)

    elif receptive_field_norm == "column":
        n = attentions.size(-1)

        # Receptive field size for each column: [n, n-1, n-2, ..., 1]
        receptive_field = torch.arange(
            n, 0, -1, dtype=dtype, device=attentions.device
        )

        # Divide each column by its receptive field size
        attentions = attentions / receptive_field[None, None, None, :]

        # Renormalize rows
        attentions = attentions / attentions.sum(dim=-1, keepdim=True)

    # Normalize the attention matrix to take into account the residual connections
    num_layers = attentions.size(0)
    hidden_states = hidden_states[:-1]  # removes output hidden states

    if proportion == "norm":
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

    elif proportion == "projection":

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


def attention_geometric_mean(
    attentions: torch.Tensor,
    pool: Literal["max", "mean"],
    epsilon: float = 0.01,
) -> torch.Tensor:
    """
    Perform element wise geometric mean over the attention matrix, with max or mean pooling over the heads.

    Input:
    attentions (torch.Tensor): Tensor of shape
    [num_layers, batch_size, num_heads, total_length, total_length]

    Output:
    torch.Tensor: Tensor of shape
    [batch_size, total_length, total_length]
    """

    num_layers = attentions.size(0)

    if pool == "max":
        attentions, _ = attentions.max(dim=2)
    else:
        attentions = attentions.mean(dim=2)

    product = (1 - epsilon) * attentions[0] + epsilon
    for layer_idx in range(1, num_layers):
        m = (1 - epsilon) * attentions[layer_idx] + epsilon
        product = product * m

    geometric_mean = product.pow(1 / num_layers)

    return geometric_mean


def attention_additive_mean(
    attentions: torch.Tensor, pool: Literal["max", "mean"]
) -> torch.Tensor:
    """
    Perform element wise additive mean over the attention matrix, with max or mean pooling over the heads.

    Input:
    attentions (torch.Tensor): Tensor of shape
    [num_layers, batch_size, num_heads, total_length, total_length]

    Output:
    torch.Tensor: Tensor of shape
    [batch_size, total_length, total_length]
    """

    if pool == "max":
        attentions, _ = attentions.max(dim=2)
    else:
        attentions = attentions.mean(dim=2)

    additive_mean = attentions.mean(dim=0)
    return additive_mean


def aggregate_attention_influence(influence: torch.Tensor) -> torch.Tensor:
    n = influence.size(-1)
    aggregated_influence = influence.sum(dim=0)
    divisors = torch.arange(
        n, 0, -1, dtype=influence.dtype, device=influence.device
    )
    aggregated_influence = aggregated_influence / divisors

    return aggregated_influence
