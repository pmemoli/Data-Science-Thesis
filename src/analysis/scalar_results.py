# Computes AUROC, Value and accuracy for a bunch of scalar metric.

from sklearn.metrics import roc_auc_score
import torch
import os

suite = "phi3-gsm-test"
tensor_path = f"src/data/runs/{suite}"
tensor_files = os.listdir(tensor_path)


def weighted_quantile(values, weights, quantile):
    """Compute weighted quantile"""
    sorted_indices = torch.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cumsum = torch.cumsum(sorted_weights, dim=0)
    cutoff = quantile * cumsum[-1]
    idx = torch.searchsorted(cumsum, cutoff)
    # Clamp idx to valid range
    idx = torch.clamp(idx, 0, len(sorted_values) - 1)
    return sorted_values[idx]


def summary_distribution_metrics(profile, influence=None):
    """
    Input:
        profile: Tensor of shape [sequence_length]
        influence: Tensor of shape [sequence_length] or None
    """

    if influence is not None:
        weights = influence / influence.sum()

        mean_value = torch.sum(profile * weights, dim=0)
        weighted_var = torch.sum(weights * (profile - mean_value) ** 2, dim=0)
        std_value = torch.sqrt(weighted_var)

        # Weighted quantiles
        q25 = weighted_quantile(profile, weights, 0.25)
        q50 = weighted_quantile(profile, weights, 0.50)
        q75 = weighted_quantile(profile, weights, 0.75)
        q90 = weighted_quantile(profile, weights, 0.90)
        q99 = weighted_quantile(profile, weights, 0.99)

        max_value = weighted_quantile(profile, weights, 1)

    else:
        mean_value = torch.mean(profile, dim=0)
        std_value = torch.std(profile, dim=0)
        q25 = torch.quantile(profile, 0.25, dim=0)
        q50 = torch.quantile(profile, 0.50, dim=0)
        q75 = torch.quantile(profile, 0.75, dim=0)
        q90 = torch.quantile(profile, 0.90, dim=0)
        q99 = torch.quantile(profile, 0.99, dim=0)

        max_value = torch.max(profile, dim=0).values

    return {
        "mean": mean_value,
        "std": std_value,
        "max": max_value,
        "q25": q25,
        "q50": q50,
        "q75": q75,
        "q90": q90,
        "q99": q99,
    }


def full_summary_metrics(profile, influence=None):
    """
    Input:
        profile: Tensor of shape [layers, sequence_length]
        influence: Tensor of shape [layers, sequence_length] or None
    """

    metrics = {}

    influence = influence[-1] if influence is not None else None

    # Get last layer metrics
    last_layer_profile = profile[-1]
    last_layer_metrics = summary_distribution_metrics(
        last_layer_profile, influence
    )
    for metric_name, metric_value in last_layer_metrics.items():
        metrics[f"last_layer_{metric_name}"] = metric_value

    # Get variance metrics
    for layer_count in range(2, 5):
        layer_profile = profile[-layer_count:]
        var_profile = torch.var(layer_profile, dim=0)

        var_metrics = summary_distribution_metrics(
            var_profile,
            influence,
        )
        for metric_name, metric_value in var_metrics.items():
            metrics[f"var_last_{layer_count}_layers_{metric_name}"] = (
                metric_value
            )

    return metrics


influence_names = [
    None,
    "rollout_proportion_norm_rfn_element_headpool_mean_max",
    "additive_mean_headpool_mean_max",
    "geometric_mean_headpool_max_eps_0.01_max",
]

positive_mean = {}
negative_mean = {}
i = 0
for file in tensor_files:
    print(f"{i}/{len(tensor_files)}")
    i += 1

    full_path = f"{tensor_path}/{file}"
    tensor = torch.load(full_path)
    for tensor_item in tensor:
        prompt_length = tensor_item["prompt_length"]
        success = tensor_item["success"]
        shannon_profile = tensor_item["full_layer_shannon_entropy"].to(
            torch.float32
        )[1:, 0]

        influences = tensor_item["attention_maps"]

        for influence_name in influence_names:
            if influence_name is not None:
                influence_tensor = influences[influence_name][:, 0]

                summary_metrics = full_summary_metrics(
                    shannon_profile,
                    influence=influence_tensor,
                )
            else:
                summary_metrics = full_summary_metrics(shannon_profile)

            for metric_name, metric_value in summary_metrics.items():
                name = f"influence_{influence_name}_{metric_name}"

                if success:
                    if name not in positive_mean:
                        positive_mean[name] = []
                    positive_mean[name].append(metric_value.item())
                else:
                    if name not in negative_mean:
                        negative_mean[name] = []
                    negative_mean[name].append(metric_value.item())

auroc_results = {}

# After your loop completes, for each metric:
for metric_name in positive_mean.keys():
    try:
        y_true = [0] * len(positive_mean[metric_name]) + [1] * len(
            negative_mean[metric_name]
        )

        # Concatenate the metric values
        y_scores = positive_mean[metric_name] + negative_mean[metric_name]

        # Esto es lo mas feo que hice en toda mi vida
        dataset_accuracy = len(positive_mean[metric_name]) / (
            len(negative_mean[metric_name]) + len(positive_mean[metric_name])
        )

        # Compute AUROC
        auroc = roc_auc_score(y_true, y_scores)
        auroc_results[metric_name] = auroc

        print(f"{metric_name}: AUROC = {auroc:.4f}")

    except Exception as e:
        print(f"Error computing AUROC for {metric_name}: {e}")


parsed_results = {
    "metrics": {
        "no_influence": {},
        "rollout_influence": {},
        "additive_influence": {},
        "geometric_influence": {},
    },
    "dataset_accuracy": dataset_accuracy,  # type: ignore
}

for influence_name in influence_names:
    influence_name_metrics = {}

    if influence_name is None:
        key = "no_influence"
    elif "rollout" in influence_name:
        key = "rollout_influence"
    elif "additive" in influence_name:
        key = "additive_influence"
    else:
        key = "geometric_influence"

    for metric_name, auroc in auroc_results.items():
        if f"{influence_name}" in metric_name:
            base_metric_name = metric_name.replace(
                f"influence_{influence_name}_", ""
            )

            influence_name_metrics[base_metric_name] = {
                "auroc": auroc,
                "mean": (
                    sum(positive_mean[metric_name])
                    + sum(negative_mean[metric_name])
                )
                / (
                    len(positive_mean[metric_name])
                    + len(negative_mean[metric_name])
                ),
            }

    parsed_results["metrics"][key] = influence_name_metrics


print(parsed_results)
