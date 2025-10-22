from typing import Dict, Any, List
from dataclasses import dataclass
from src.metrics.attention import (
    attention_rollout,
    influence,
    attention_geometric_mean,
    attention_additive_mean,
    receptive_field_mean,
    max_aggregation,
    mean_aggregation,
    last_token_aggregation,
)
import torch


@dataclass
class AttentionConfig:
    """Configuration for a single attention computation"""

    name: str
    method: str
    kwargs: Dict[str, Any]


AGGREGATION_FUNCTIONS = {
    "receptive_field_mean": receptive_field_mean,
    "max": max_aggregation,
    "mean": mean_aggregation,
    "last_token": last_token_aggregation,
}


def compute_attention_maps(
    attentions: torch.Tensor,
    hidden_states: torch.Tensor,
    attention_outputs: torch.Tensor,
    configs: List[AttentionConfig] | None = None,
    aggregations: List[str] | None = None,
) -> Dict[str, torch.Tensor]:

    if configs is None:
        configs = get_default_configs()

    if aggregations is None:
        aggregations = list(AGGREGATION_FUNCTIONS.keys())

    attention_maps = {}

    for config in configs:
        if config.method == "rollout":
            result = attention_rollout(attentions, **config.kwargs)
        elif config.method == "influence":
            result = influence(
                attentions, hidden_states, attention_outputs, **config.kwargs
            )
        elif config.method == "geometric_mean":
            result = attention_geometric_mean(attentions, **config.kwargs)
        elif config.method == "additive_mean":
            result = attention_additive_mean(attentions, **config.kwargs)
        else:
            raise ValueError(f"Unknown method: {config.method}")

        for agg_name in aggregations:
            if agg_name not in AGGREGATION_FUNCTIONS:
                raise ValueError(f"Unknown aggregation: {agg_name}")

            agg_func = AGGREGATION_FUNCTIONS[agg_name]
            aggregated = agg_func(result)

            key = f"{config.name}_{agg_name}"
            attention_maps[key] = aggregated.half().cpu()

    return attention_maps


def get_default_configs() -> List[AttentionConfig]:
    configs = []

    for pool in ["mean", "max"]:
        pool_suffix = f"pool_{pool}"
        for rfn in ["element", "column", None]:
            rfn_suffix = f"rfn_{rfn}" if rfn else "nrfn"

            configs.extend(
                [
                    AttentionConfig(
                        method="rollout",
                        name=f"rollout_05_05_{rfn_suffix}_{pool_suffix}",
                        kwargs={
                            "residual_stream_proportion": 0.5,
                            "attention_output_proportion": 0.5,
                            "receptive_field_norm": rfn,
                            "pool": pool,
                        },
                    ),
                    AttentionConfig(
                        method="rollout",
                        name=f"rollout_09_01_{rfn_suffix}_{pool_suffix}",
                        kwargs={
                            "residual_stream_proportion": 0.9,
                            "attention_output_proportion": 0.1,
                            "receptive_field_norm": rfn,
                            "pool": pool,
                        },
                    ),
                ]
            )

            for prop in ["norm", "projection"]:
                configs.append(
                    AttentionConfig(
                        method="influence",
                        name=f"influence_{prop}_{rfn_suffix}_{pool_suffix}",
                        kwargs={
                            "proportion": prop,
                            "receptive_field_norm": rfn,
                            "pool": pool,
                        },
                    )
                )

        for epsilon in [0.001, 0.01, 0.1]:
            epsilon_suffix = f"eps_{epsilon}"
            for shift in [0.0, 1.0]:
                shift_suffix = f"shift_{shift}"
                configs.append(
                    AttentionConfig(
                        method="geometric_mean",
                        name=f"geometric_mean_{pool_suffix}_{epsilon_suffix}_{shift_suffix}",
                        kwargs={
                            "pool": pool,
                            "epsilon": epsilon,
                            "shift": shift,
                        },
                    )
                )

        configs.append(
            AttentionConfig(
                method="additive_mean",
                name=f"additive_mean_{pool_suffix}",
                kwargs={"pool": pool},
            )
        )

    return configs
