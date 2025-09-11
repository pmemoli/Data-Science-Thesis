import torch.nn.functional as F
from typing import Literal
import torch

eps = 1e-8

# Early exit based metrics
EarlyExitUQMetric = Literal[
    "state_mean_exit_layer",
    "softmax_mean_exit_layer",
]

def early_exit_uq(
    hidden_states, # [layer, batch_size, sequence_length, hidden_size] 
    lm_head,
    threshold: float,
    metric_name: EarlyExitUQMetric,
):
    """
    Compute uncertainty metrics based on the layer at which the token stabilizes, and can early exit with some confidence.

    Input:
        hidden_states: tuple of tensors from the model, each of shape
                       [layer, batch_size, sequence_length, hidden_size]
        lm_head: the language model head to project hidden states to vocab size
        threshold: float between 0 and 1 indicating the threshold for early exit
        metric_name: one of "shannon_entropy", "predictive_entropy", "negative_log_likelihood"

    Output:
        token_uq: tensor of shape [batch_size, sequence_length] with uncertainty scores
    """

    with torch.no_grad():
        layer_amount = hidden_states.shape[0]

        # Contains the uncertainty of each token at each layer
        layer_uq_tensor = torch.zeros(
            hidden_states.shape[0], # layers 
            hidden_states.shape[1], # batch size
            hidden_states.shape[2], # sequence length
        ).to("cuda:0")

        for layer_idx in range(layer_amount - 1):
            layer_states = hidden_states[layer_idx]
            next_layer_states = hidden_states[layer_idx + 1]

            if metric_name == "state_mean_exit_layer":
                difference = 1 - F.cosine_similarity(
                    layer_states,
                    next_layer_states,
                    dim=-1,
                    eps=1e-8
                )

            elif metric_name == "softmax_mean_exit_layer":
                layer_distribution = F.softmax(lm_head(layer_states), dim=-1)
                top_2_token_prob, _ = torch.topk(layer_distribution, 2, dim=-1)

                difference = 1 - (top_2_token_prob[:, :, 0] - top_2_token_prob[:, :, 1])

            # [layers, batch_size, sequence_length]
            layer_uq_tensor[layer_idx] = difference > threshold

        # [batch_size, sequence_length]
        token_uq = torch.argmin(layer_uq_tensor, dim=0)
    
    return token_uq
