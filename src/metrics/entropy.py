import torch

epsilon = 1e-9


def predictive_entropy(token_probabilities: torch.Tensor) -> torch.Tensor:
    """
    Computes the normalized surprisal of a sequence of tokens.
    token_probabilities must be of dimension [batch_size, sequence_length].
    """

    token_log_probabilities = (token_probabilities + epsilon).log()
    sequence_entropy = -torch.mean(token_log_probabilities, dim=-1)

    return sequence_entropy


def shannon_entropy(token_distribution: torch.Tensor) -> torch.Tensor:
    """
    Computes the average Shannon entropy for each generated token distribution.
    token_distribution must be of dimension [batch_size, sequence_length, vocab_size].
    """

    token_log_distribution = (token_distribution + epsilon).log()

    token_entropy = -torch.sum(
        token_distribution * token_log_distribution, dim=-1
    )
    sequence_entropy = torch.mean(token_entropy, dim=-1)

    return sequence_entropy
