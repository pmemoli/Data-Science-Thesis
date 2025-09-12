import torch.nn.functional as F
from typing import Literal
import torch

eps = 1e-8

def kl_divergence(probs_p: torch.Tensor, probs_q: torch.Tensor) -> torch.Tensor:
    """
    Input: probs_p [batch_size, sequence_length, vocab_size], probs_q idem
    Output: [batch_size, sequence_length]
    """

    probs_p = probs_p.clamp(min=eps)
    probs_q = probs_q.clamp(min=eps)
    kl = (probs_p * (probs_p.log() - probs_q.log())).sum(dim=-1)

    return kl

LayerEvolutionUQMetric = Literal[
    "layer_evolution_mean_kl_divergence", 
    "layer_evolution_var_kl_divergence", 
    "layer_evolution_mean_shannon_entropy", 
    "layer_evolution_var_shannon_entropy"
]

def layer_evolution_uq(
    hidden_states, # [num_layers, batch_size, generated_length, hidden_size]
    lm_head, 
    metric_name: LayerEvolutionUQMetric,
    layers_from_end: int = 1,
):
    """
    Compute uncertainty metrics based on layer output evolution.

    Input:
        hidden_states: tuple of tensors from the model, each of shape
                       [layer, batch_size, sequence_length, hidden_size]
        lm_head: the language model head to project hidden states to vocab size
        metric_name: one of "shannon_entropy", "predictive_entropy", "negative_log_likelihood"

    Output:
        token_uq: tensor of shape [batch_size, sequence_length] with uncertainty scores
    """

    with torch.no_grad():
        last_layer_distribution = F.softmax(lm_head(hidden_states[-1]), dim=-1)

        layer_amount = hidden_states.shape[0]

        layer_uq_tensor = torch.zeros(
            hidden_states.shape[0], # layers 
            hidden_states.shape[1], # batch size
            hidden_states.shape[2], # sequence length
        ).to("cuda:0")

        for layer_states_idx in range(layer_amount - layers_from_end, layer_amount - 1):
            layer_states = hidden_states[layer_states_idx]

            layer_distribution = F.softmax(lm_head(layer_states), dim=-1)

            if "kl_divergence" in metric_name:
                layer_uq = kl_divergence(
                    layer_distribution, 
                    last_layer_distribution
                )
            elif "shannon_entropy" in metric_name:
                layer_uq = -torch.sum(
                    layer_distribution * torch.log(layer_distribution + eps), 
                    dim=-1
                )

            layer_uq_tensor[layer_states_idx, :, :] = layer_uq # type: ignore

        if "mean" in metric_name:
            token_uq = torch.mean(layer_uq_tensor, dim=0)
        elif "var" in metric_name:
            token_uq = torch.var(layer_uq_tensor, dim=0)

    return token_uq # type: ignore
