import torch

epsilon = 1e-9


def predictive_entropy(
    token_probabilities: torch.Tensor,
    sequence_length: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Computes the normalized negative log likelihood of a sampled sequence of tokens.

    token_probabilities must be of dimension [batch_size, sequence_length].

    sequence_length must be of dimension [batch_size].
    """
    # This tensor is derived from an input, so it's already on the correct device
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
        [batch_size, layer_amount, sequence_length, vocab_size].

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
