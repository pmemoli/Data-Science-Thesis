import torch

epsilon = 1e-9


def predictive_entropy(
    token_probabilities: torch.Tensor,
    sequence_length: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Computes the normalized negative log likelihood of a sampled sequence of tokens:
    - log(P(S | X)) / sequence_length

    token_probabilities must be of dimension [batch_size, sequence_length].

    sequence_length must be of dimension [batch_size].
    """

    token_nll = -(token_probabilities + epsilon).log()

    # Mask to ignore the padded tokens
    mask = torch.arange(token_probabilities.size(1), device=device).unsqueeze(
        0
    ) < sequence_length.unsqueeze(1)
    token_nll = torch.where(
        mask, token_nll, torch.tensor(torch.nan, device=device)
    )

    normalized_sequence_nll = torch.nanmean(token_nll, dim=-1)

    return normalized_sequence_nll


def shannon_entropy(
    token_distribution: torch.Tensor,
    sequence_length: torch.Tensor,
    layer=-1,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Computes the average Shannon entropy for each generated token distribution.

    token_distribution must be of dimension:
        [batch_size, layer, sequence_length, top_k].

    sequence_length must be of dimension [batch_size].
    """

    layer_distribution = token_distribution[:, layer, :, :]
    token_log_distribution = (layer_distribution + epsilon).log()
    token_entropy = -torch.sum(
        layer_distribution * token_log_distribution, dim=-1
    )

    # Mask to ignore the padded tokens
    position_index = torch.arange(
        token_entropy.size(1), device=device
    ).unsqueeze(0)
    sequence_length = sequence_length.unsqueeze(1)

    mask = position_index < sequence_length

    token_entropy = torch.where(
        mask, token_entropy, torch.tensor(torch.nan, device=device)
    )
    sequence_entropy = torch.nanmean(token_entropy, dim=-1)

    return sequence_entropy


def attention_entropy(attentions: torch.Tensor) -> torch.Tensor:
    """
    Computes the average attention entropy for each token and head

    attentions must be of type:
        [batch_size, layer, sequence_length, top_k].

    sequence_length must be of dimension [batch_size].

    I am not taking into account the variable generated sequence length between batches!
    """

    token_entropy_list = []  # list of seq_len tensors [batch_size]
    for token_attention in attentions:
        token_attention_log = -token_attention.log()
        token_head_entropy = (token_attention_log * token_attention).mean(
            dim=2
        )
        token_mean_entropy = token_head_entropy.mean(dim=1)

        token_entropy_list.append(token_mean_entropy)

    head_attention_entropy = torch.stack(token_entropy_list, dim=1)
    attention_entropy = torch.mean(head_attention_entropy, dim=1)

    return attention_entropy
