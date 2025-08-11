import torch

epsilon = 1e-9


def predictive_entropy(
    token_probabilities: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the normalized negative log likelihood of a sampled sequence of tokens:
    - log(P(S | X)) / sequence_length

    token_probabilities must be of dimension
        [batch_size, sequence_length].
    """

    token_nll = -(token_probabilities + epsilon).log()
    normalized_sequence_nll = torch.nanmean(token_nll, dim=-1)

    return normalized_sequence_nll


def shannon_entropy(
    token_distribution: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the average Shannon entropy for each generated token distribution.

    token_distribution must be of dimension:
        [batch_size, sequence_length, top_k].
    """

    token_log_distribution = (token_distribution + epsilon).log()
    token_entropy = -torch.sum(
        token_distribution * token_log_distribution, dim=-1
    )
    sequence_entropy = torch.nanmean(token_entropy, dim=-1)

    return sequence_entropy


def attention_entropy(attentions: torch.Tensor) -> torch.Tensor:
    """
    Computes the average attention entropy for each token and head

    attentions must be of type:
        [batch_size, num_heads, output_sequence_length, full_sequence_length].
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
