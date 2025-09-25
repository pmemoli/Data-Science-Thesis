from .attention import attention_rollout, influence
import gc
import torch

eps = 1e-8

weighting_options = [
    "prob",
    "attention_rollout",
    "attention_influence_norm",
    "attention_influence_angle",
    "attention_influence_projection",
]


def sequence_ensemble(
    metric: torch.Tensor,  # [batch_size, sequence_length]
    last_layer_distribution: torch.Tensor,
    attentions: torch.Tensor,
    attention_outputs: torch.Tensor,
    hidden_states: torch.Tensor,
    pooling_ratio: float = 1,
    prompt_length: int = 0,
    weighting: None | str = None,
    quantile: float | None = None,
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

    with torch.no_grad():
        metric = metric[:, prompt_length:]

        if not weighting:
            pass

        elif weighting == "entropy":
            weights = -torch.sum(
                last_layer_distribution
                * torch.log(last_layer_distribution + 1e-8),
                dim=-1,
            )

        elif weighting == "prob":
            max_probs, _ = torch.max(last_layer_distribution, dim=-1)
            weights = 1 - max_probs

        elif "attention_rollout" in weighting:
            rollout = attention_rollout(attentions, 0.96, 0.04)
            weights = torch.mean(rollout, dim=1)

        elif "attention_influence" in weighting:
            if "norm" in weighting:
                mode = "norm"
            elif "angle" in weighting:
                mode = "angle"
            else:
                mode = "projection"

            infl = influence(
                attentions, hidden_states, attention_outputs, mode
            )
            weights = torch.mean(infl, dim=1)

        pool_amount = max(1, int(pooling_ratio * metric.shape[1]))

        # Weight the metric
        if weighting:
            weights = weights[:, prompt_length:]

            # If weighting, choose the top k according to the weights
            top_k_weights, top_k_ids = torch.topk(weights, pool_amount, dim=-1)
            top_k_values = torch.gather(metric, dim=-1, index=top_k_ids)

        else:
            top_k_values, top_k_ids = torch.topk(metric, pool_amount, dim=-1)

        if quantile:
            result = torch.quantile(top_k_values, quantile, dim=-1)

        else:
            if weighting:
                top_k_values = (
                    top_k_values
                    * top_k_weights
                    / torch.sum(top_k_weights, dim=-1)
                )

            else:
                top_k_values = top_k_values / pool_amount

            result = torch.sum(top_k_values, dim=-1)

    torch.cuda.empty_cache()
    gc.collect()

    return result
