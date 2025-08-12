# These only really work with batch_size 1, but they are silly inexpensive so its fine

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

    token_log_distribution = -(token_distribution + epsilon).log()
    token_entropy = torch.sum(
        token_distribution * token_log_distribution, dim=-1
    )
    sequence_entropy = torch.mean(token_entropy, dim=-1)

    return sequence_entropy


def attention_entropy(attentions: torch.Tensor) -> torch.Tensor:
    """
    Computes the average attention entropy for each token and head

    attentions must be of type:
        [batch_size, num_heads, output_sequence_length, full_sequence_length].
    """
    attention_entropy = torch.zeros(attentions.size(0))
    for i in range(attentions.size(2)):
        token_attentions = attentions[
            :, :, i, : attentions.size(3) - attentions.size(2) + i + 1
        ]
        nlog_token_attentions = -(token_attentions + 1e-7).log()
        token_attention_entropy = torch.sum(
            nlog_token_attentions * token_attentions, dim=-1
        )
        mean_token_attention_entropy = torch.mean(
            token_attention_entropy, dim=1
        )

        normalized_token_ae = mean_token_attention_entropy
        attention_entropy += normalized_token_ae

    attention_entropy = attention_entropy / attentions.size(2)

    return attention_entropy
