import torch.nn.functional as F
from typing import Literal
import torch

eps = 1e-8

# Last layer distribution based metrics
LogitsUQMetric = Literal[
    "logits_shannon_entropy",
    "logits_predictive_entropy",
    "logits_negative_log_likelihood"
]

def logits_uq(
    hidden_states, # [layer, batch_size, sequence_length, hidden_size] 
    lm_head, 
    sequences: torch.Tensor,
    metric_name: LogitsUQMetric,
):
    """
    Compute uncertainty metrics based on the last layer distribution.

    Input:
        hidden_states: tuple of tensors from the model, each of shape
                       [layer, batch_size, sequence_length, hidden_size]
        lm_head: the language model head to project hidden states to vocab size
        sequences: tensor of shape [batch_size, sequence_length] with token ids
        metric_name: one of "shannon_entropy", "predictive_entropy", "negative_log_likelihood"

    Output:
        token_uq: tensor of shape [batch_size, sequence_length] with uncertainty scores
    """

    with torch.no_grad():
        last_layer_distribution = F.softmax(lm_head(hidden_states[-1]), dim=-1)

        if metric_name == "logits_shannon_entropy":
            token_uq = -torch.sum(
                last_layer_distribution * torch.log(last_layer_distribution + eps), 
                dim=-1
            )

        elif metric_name == "logits_negative_log_likelihood":
            token_uq = -torch.log(torch.gather(
                last_layer_distribution, 
                dim=-1, 
                index=sequences.unsqueeze(-1)
            ).squeeze(-1) + eps)

        elif metric_name == "logits_predictive_entropy":
            selected_token_probs = torch.gather(
                last_layer_distribution, 
                dim=-1, 
                index=sequences.unsqueeze(-1)
            ).squeeze(-1)

            token_uq = -selected_token_probs * torch.log(selected_token_probs + eps)

    return token_uq
