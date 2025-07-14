import torch

epsilon = 1e-9


def predictive_entropy(
    token_probabilities: torch.Tensor, sequence_length: torch.Tensor
) -> torch.Tensor:
    """
    Computes the normalized negative log likelihood of a sequence of tokens.
    token_probabilities must be of dimension [batch_size, sequence_length].
    sequence_length must be of dimension [batch_size].
    Ignores zero-padded positions in the mean calculation.
    """

    token_nll = -(token_probabilities + epsilon).log()

    # Mask to ignore the padded tokens
    mask = torch.arange(token_probabilities.size(1)).unsqueeze(
        0
    ) < sequence_length.unsqueeze(1)
    token_nll = torch.where(mask, token_nll, torch.nan)

    normalized_sequence_nll = torch.nanmean(token_nll, dim=-1)

    return normalized_sequence_nll


def shannon_entropy(
    token_distribution: torch.Tensor, sequence_length: torch.Tensor
) -> torch.Tensor:
    """
    Computes the average Shannon entropy for each generated token distribution.
    token_distribution must be of dimension [batch_size, sequence_length, vocab_size].
    sequence_length must be of dimension [batch_size].
    Ignores zero-padded positions in the mean calculation.
    """

    token_log_distribution = (token_distribution + epsilon).log()

    token_entropy = -torch.sum(
        token_distribution * token_log_distribution, dim=-1
    )

    # Mask to ignore the padded tokens
    mask = torch.arange(token_entropy.size(1)).unsqueeze(
        0
    ) < sequence_length.unsqueeze(1)
    token_entropy = torch.where(mask, token_entropy, torch.nan)

    sequence_entropy = torch.nanmean(token_entropy, dim=-1)

    return sequence_entropy
