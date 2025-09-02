# White box metrics are exclusively based on token distribution, and can be computed in a single pass.

import torch
import torch.nn.functional as F
from typing import Literal
import gc

eps = 1e-8

# Reshape hidden states from model.generate()
def hidden_state_reshape(hidden_states: tuple[tuple[torch.Tensor]]) -> torch.Tensor:
    """
    Reshape the hidden states from the model output to a tensor of shape:
    [num_layers, batch_size, generated_length, hidden_size]
    """
    num_steps = len(hidden_states)

    step_tensors = []

    for step_idx in range(0, num_steps):
        step_hidden_states = hidden_states[step_idx]
        step_hidden_states = torch.stack(step_hidden_states, dim=0) 

        if step_idx != 0:
            step_tensors.append(step_hidden_states)
        else:
            step_tensors.append(step_hidden_states[:, :, -1, :].unsqueeze(2))

    return torch.cat(step_tensors, dim=2) # type: ignore


# Token-pooling function
def pool_uq_tokens(
    metric: torch.Tensor, 
    last_layer_distribution: torch.Tensor,
    sequences: torch.Tensor,
    pad_token_id: int,
    pooling_ratio: float = 1,
    weighting: None | Literal["entropy", "prob"] = None,
):
    """
    Pool the uncertainty metric over the generated tokens based on different criterions.

    -metric: [batch_size, sequence_length]
    -last_layer_distribution: [batch_size, sequence_length, vocab_size]
    -output_mask: None or [batch_size, sequence_length] with 1 for valid tokens and 0 for padding/eos
    -pooling_ratio: float between 0 and 1 indicating the ratio of tokens to pool
    -weighting: None or "entropy" or "prob" to weight the metric by the token

    returns [batch_size]
    """

    output_mask = (sequences != pad_token_id).float() 

    print(output_mask)

    with torch.no_grad():
        # Normalize the divergence importance per token
        if weighting == "entropy":
            weights = -torch.sum(last_layer_distribution * torch.log(last_layer_distribution + 1e-8), dim=-1)

        elif weighting == "prob":
            max_probs, _ = torch.max(last_layer_distribution, dim=-1)
            weights = 1 - max_probs
        
        if weighting:
            metric = weights * metric # type: ignore

        # Pool the top largest divergences based on the ratio
        pool_amount = max(1, int(pooling_ratio * metric.shape[1]))
        top_k_values, top_k_ids = torch.topk(metric, pool_amount, dim=-1)

        if output_mask is not None:
            selected_mask_values = torch.gather(
                output_mask, 
                dim=-1, 
                index=top_k_ids
            )
            top_k_values = torch.where(selected_mask_values == 0, torch.nan, top_k_values)

        # Mean over remaining pool (ignoring padding/eos)
        result = torch.nanmean(top_k_values, dim=-1) 

    torch.cuda.empty_cache()
    gc.collect()

    return result


# Last layer distribution based metrics
LastLayerDistributionUQMetric = Literal[
    "shannon_entropy",
    "predictive_entropy",
    "negative_log_likelihood"
]

def last_layer_distribution_uq(
    hidden_states, # [layer, batch_size, sequence_length, hidden_size] 
    lm_head, 
    sequences: torch.Tensor,
    pad_token_id: int,
    metric_name: LastLayerDistributionUQMetric,
    pooling_ratio=1,
    weighting: None | Literal["entropy", "prob"] = None,
):
    with torch.no_grad():
        last_layer_distribution = F.softmax(lm_head(hidden_states[-1]), dim=-1)

        if metric_name == "shannon_entropy":
            token_uq = -torch.sum(
                last_layer_distribution * torch.log(last_layer_distribution + eps), 
                dim=-1
            )

        elif metric_name == "negative_log_likelihood":
            token_uq = -torch.log(torch.gather(
                last_layer_distribution, 
                dim=-1, 
                index=sequences.unsqueeze(-1)
            ).squeeze(-1) + eps)

        elif metric_name == "predictive_entropy":
            selected_token_probs = torch.gather(
                last_layer_distribution, 
                dim=-1, 
                index=sequences.unsqueeze(-1)
            ).squeeze(-1)

            token_uq = -selected_token_probs * torch.log(selected_token_probs + eps)

        result = pool_uq_tokens(
            token_uq, # type: ignore
            last_layer_distribution,
            sequences,
            pad_token_id,
            pooling_ratio,
            weighting
        )

    return result

# Layer evolution based metrics
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
    "mean_kl_divergence", 
    "var_kl_divergence", 
    "mean_shannon_entropy", 
    "var_shannon_entropy"
]

def layer_evolution_uq(
    hidden_states, # [num_layers, batch_size, generated_length, hidden_size]
    lm_head, 
    sequences: torch.Tensor,
    pad_token_id: int,
    metric_name: LayerEvolutionUQMetric,
    layers_from_end: int = 1,
    pooling_ratio:float = 1,
    weighting: None | Literal["entropy", "prob"] = None,
):
    hidden_states.to("cpu")
    lm_head.to("cpu")

    with torch.no_grad():
        last_layer_distribution = F.softmax(lm_head(hidden_states[-1]), dim=-1)

        layer_amount = hidden_states.shape[0]

        layer_uq_tensor = torch.zeros(
            hidden_states.shape[0], # layers 
            hidden_states.shape[1], # batch size
            hidden_states.shape[2], # sequence length
        ).to("cpu")

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

    result = pool_uq_tokens(
        token_uq, # type: ignore
        last_layer_distribution,
        sequences,
        pad_token_id,
        pooling_ratio,
        weighting,
    )

    torch.cuda.empty_cache()
    gc.collect()

    return result 

# Early exit based metrics
EarlyExitUQMetric = Literal[
    "state_mean_exit_layer",
    "softmax_mean_exit_layer",
]

def early_exit_uq(
    hidden_states, # [layer, batch_size, sequence_length, hidden_size] 
    lm_head,
    threshold: float,
    sequences: torch.Tensor,
    pad_token_id: int,
    metric_name: EarlyExitUQMetric,
    pooling_ratio=1,
    weighting: None | Literal["entropy", "prob"] = None,
):
    with torch.no_grad():
        last_layer_distribution = F.softmax(lm_head(hidden_states[-1]), dim=-1)

        layer_amount = hidden_states.shape[0]
        layer_uq_tensor = torch.zeros(
            hidden_states.shape[0], # layers 
            hidden_states.shape[1], # batch size
            hidden_states.shape[2], # sequence length
        ).to("cpu")

        for layer_idx in range(len(layer_amount) - 1):
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
                top_token_prob, _ = torch.max(layer_distribution, dim=-1) # [batch_size, sequence_length]

                next_layer_distribution = F.softmax(lm_head(next_layer_states), dim=-1)
                next_layer_top_token_prob, _ = torch.max(next_layer_distribution, dim=-1) # [batch_size, sequence_length]

                difference = torch.abs(top_token_prob - next_layer_top_token_prob)

            # [layers, batch_size, sequence_length]
            layer_uq_tensor[layer_idx] = (difference < threshold).float()

        token_uq = torch.argmax(layer_uq_tensor, dim=0)

        result = pool_uq_tokens(
            token_uq, # type: ignore
            last_layer_distribution,
            sequences,
            pad_token_id,
            pooling_ratio,
            weighting
        )

    return result

