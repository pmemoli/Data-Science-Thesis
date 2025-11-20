from sklearn.metrics import roc_auc_score
import torch
import os

suite = "phi3-gsm-test"
tensor_path = f"src/data/runs/{suite}"
tensor_files = os.listdir(tensor_path)


def mean_profile_variance(profile, layers_from_end=1):
    selected_profile = profile[-layers_from_end:]
    var_profile = torch.var(selected_profile, dim=0)
    mean_profile = torch.mean(var_profile, dim=0)
    return mean_profile


def early_exit(profile, threshold):
    exit_layers = []
    for item_idx in range(profile.shape[1]):
        for layer in range(profile.shape[0] - 1):
            profile_value = profile[layer, item_idx]

            if profile_value > threshold:
                exit_layers.append(layer + 1)
                break

    return sum(exit_layers) / len(exit_layers)


positive_mean = {
    "mean_se": [],
    "mean_nll": [],
    "mean_pe": [],
}
negative_mean = {
    "mean_se": [],
    "mean_nll": [],
    "mean_pe": [],
}
i = 0
for file in tensor_files:
    print(f"Processing file {i+1}/{len(tensor_files)}: {file}")
    i += 1

    full_path = f"{tensor_path}/{file}"
    tensor = torch.load(full_path)
    for tensor_item in tensor:
        prompt_length = tensor_item["prompt_length"]
        success = tensor_item["success"]
        shannon_profile = tensor_item["full_layer_shannon_entropy"][
            :, 0, prompt_length:
        ]
        selected_prob_profile = tensor_item["full_layer_selected_prob"][
            :, 0, prompt_length:
        ]
        selected_log_prob_profile = -torch.log(selected_prob_profile + 1e-8)

        # Mean SE, PE, NLL (last layer)
        mean_se = torch.mean(shannon_profile[-1], dim=0)
        log_probs = selected_log_prob_profile[-1]
        probs = selected_prob_profile[-1]

        mean_nll = torch.mean(log_probs, dim=0)
        mean_pe = torch.mean(probs * log_probs, dim=0)

        if success:
            positive_mean["mean_se"].append(mean_se.item())
            positive_mean["mean_nll"].append(mean_nll.item())
            positive_mean["mean_pe"].append(mean_pe.item())
        else:
            negative_mean["mean_se"].append(mean_se.item())
            negative_mean["mean_nll"].append(mean_nll.item())
            negative_mean["mean_pe"].append(mean_pe.item())

        # Var SE, PE, NLL
        for layers in range(2, 11):
            var_se = mean_profile_variance(
                shannon_profile, layers_from_end=layers
            )
            var_nll = mean_profile_variance(
                selected_log_prob_profile, layers_from_end=layers
            )
            var_pe = mean_profile_variance(
                selected_prob_profile * selected_log_prob_profile,
                layers_from_end=layers,
            )

            if success:
                key_se = f"var_se_last_{layers}"
                key_nll = f"var_nll_last_{layers}"
                key_pe = f"var_pe_last_{layers}"
                if key_se not in positive_mean:
                    positive_mean[key_se] = []
                    positive_mean[key_nll] = []
                    positive_mean[key_pe] = []

                positive_mean[key_se].append(var_se.item())
                positive_mean[key_nll].append(var_nll.item())
                positive_mean[key_pe].append(var_pe.item())
            else:
                key_se = f"var_se_last_{layers}"
                key_nll = f"var_nll_last_{layers}"
                key_pe = f"var_pe_last_{layers}"
                if key_se not in negative_mean:
                    negative_mean[key_se] = []
                    negative_mean[key_nll] = []
                    negative_mean[key_pe] = []

                negative_mean[key_se].append(var_se.item())
                negative_mean[key_nll].append(var_nll.item())
                negative_mean[key_pe].append(var_pe.item())

# After your loop completes, for each metric:
for metric_name in positive_mean.keys():
    # Create labels: 1 for positive (success), 0 for negative (failure)
    y_true = [0] * len(positive_mean[metric_name]) + [1] * len(
        negative_mean[metric_name]
    )

    # Concatenate the metric values
    y_scores = positive_mean[metric_name] + negative_mean[metric_name]

    # Compute AUROC
    auroc = roc_auc_score(y_true, y_scores)
    print(f"{metric_name}: AUROC = {auroc:.4f}")
