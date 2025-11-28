# Stores token-level activations and outputs during generation

import torch
import argparse
import os


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


def summary_distribution_metrics(profile, influence):
    """
    Input:
        profile: Tensor of shape [sequence_length]
        influence: Tensor of shape [sequence_length] or None
    """

    weights = influence / influence.sum()

    mean_value = torch.sum(profile * weights, dim=0)
    weighted_var = torch.sum(weights * (profile - mean_value) ** 2, dim=0)
    std_value = torch.sqrt(weighted_var)

    # Weighted quantiles
    q25 = weighted_quantile(profile, weights, 0.25)
    q50 = weighted_quantile(profile, weights, 0.50)
    q75 = weighted_quantile(profile, weights, 0.75)

    max_value = weighted_quantile(profile, weights, 1)

    return {
        "mean": mean_value,
        "std": std_value,
        "max": max_value,
        "q25": q25,
        "q50": q50,
        "q75": q75,
    }


def compute_features(profile, influence):
    """
    Input:
        profile: Tensor of shape [layers, sequence_length]
        influence: Tensor of shape [sequence_length]
    """

    feature_list = []

    for i in range(6):
        layer_idx = -(i + 1)
        layer_profile = profile[layer_idx]
        layer_metrics = summary_distribution_metrics(layer_profile, influence)

        metric_tensor = torch.tensor(
            [
                layer_metrics["mean"],
                layer_metrics["std"],
                layer_metrics["max"],
            ]
        )
        feature_list.append(metric_tensor)

    metrics = torch.cat(feature_list, dim=0)

    return metrics


def run(
    suite: str,
    result_path: str,
):
    file_path = f"{result_path}"
    os.makedirs(file_path, exist_ok=True)

    tensor_path = f"src/data/runs/{suite}"
    tensor_files = os.listdir(tensor_path)

    print(f"Procesando {len(tensor_files)} archivos...")
    feature_list = []
    for i, file in enumerate(tensor_files):
        if i % 100 == 0:
            print(f"{i}/{len(tensor_files)}")

        full_path = f"{tensor_path}/{file}"
        tensor = torch.load(full_path)

        for tensor_item in tensor:
            shannon_entropies = tensor_item["full_layer_shannon_entropy"].to(
                torch.float32
            )[1:, 0]
            influence = tensor_item["attention_maps"][
                "rollout_proportion_norm_rfn_element_headpool_mean_max"
            ][-1, 0]

            print(shannon_entropies[-2])

            features = compute_features(shannon_entropies, influence)
            feature_list.append(
                {
                    "features:": features,
                    "success": tensor_item["success"],
                }
            )

    return

    save_path = f"{file_path}/{suite}_features.pt"
    torch.save(feature_list, save_path)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation on language models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--suite", type=str, required=True, help="Name of the evaluation suite"
    )

    parser.add_argument(
        "--result_path", type=str, required=True, help="Path to save results"
    )

    return parser.parse_args()


def main():
    """Main entry point for command line execution."""
    try:
        args = parse_arguments()

        print("\n with the following configuration:\n")
        print(f"  Suite: {args.suite}")
        print(f"  Results will be saved to: {args.result_path}")
        print("-" * 30)

        run(
            suite=args.suite,
            result_path=args.result_path,
        )

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"Error running benchmark: {str(e)}")
        raise


if __name__ == "__main__":
    main()
