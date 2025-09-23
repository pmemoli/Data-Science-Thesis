from .token_uq import early_exit_uq, logits_uq, layer_evolution_uq
from .sequence_ensemble import sequence_ensemble
from sklearn.metrics import roc_auc_score
from typing import Literal
import torch
import json
import hashlib


def hash_result(question, response):
    hasher = hashlib.sha256()
    hasher.update(question.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(response.encode("utf-8"))
    return hasher.hexdigest()


Metric = Literal[
    "early_exit_state_mean_exit_layer",
    "early_exit_softmax_mean_exit_layer",
    "logits_shannon_entropy",
    "logits_predictive_entropy",
    "logits_negative_log_likelihood",
    "layer_evolution_mean_kl_divergence",
    "layer_evolution_var_kl_divergence",
    "layer_evolution_mean_shannon_entropy",
    "layer_evolution_var_shannon_entropy",
]

WeightingMethod = Literal[
    "raw_mean",
    "entropy",
    "prob",
    "attention_rollout",
    "attention_influence_norm",
    "attention_influence_angle",
    "attention_influence_projection",
]

AggregationMethod = Literal[
    "all_sequence",
    "5%_sequence",
    "10%_sequence",
    "20%_sequence",
    "40%_sequence",
]


def compute_uq_auroc_grid(
    datafile: str,
    model,
    tokenizer,
    metrics: list[Metric] = [],
    aggregation_methods: list[AggregationMethod] = [],
    weighting_methods: list[WeightingMethod] = [],
):

    if len(metrics) == 0:
        metrics = list(Metric.__args__)
    if len(aggregation_methods) == 0:
        aggregation_methods = list(AggregationMethod.__args__)
    if len(weighting_methods) == 0:
        weighting_methods = list(WeightingMethod.__args__)

    # Load tensors
    data = torch.load(datafile, map_location="cpu", mmap=True)

    # Load evaluations
    eval_file = datafile.replace(".pt", ".json").replace(
        "validation/", "validation/evaluations_"
    )
    with open(eval_file, "r") as f:
        evaluations = json.load(f)

    lm_head = model.lm_head

    dataset_metrics = []

    last_layer_distribution = torch.zeros(0)

    completion = 0
    for item in data:
        print(f"Processing item {completion+1}/{len(data)}")

        hidden_states = item["hidden_states"].to(model.device)
        attention_outputs = item["attention_outputs"].to(model.device)
        attentions = item["attentions"].to(model.device)

        prompt_length = item["prompt_length"]

        sequences = item["sequences"].to(model.device)

        with torch.no_grad():
            # last_layer_distribution = torch.softmax(
            #     lm_head(hidden_states[-1]), dim=-1
            # )

            grid = {metric: {} for metric in metrics}
            grid["success"] = evaluations[
                hash_result(item["prompt"], item["generation"])
            ]

            for metric in metrics:
                for agg in aggregation_methods:
                    pooling_ratio = 1.0

                    digit = agg.split("%")[0]
                    if digit.isdigit():
                        pooling_ratio = int(digit) / 100.0

                    for weight in weighting_methods:
                        weight_param = weight

                        if "raw_mean" in weight:
                            weight_param = None

                        if "early_exit" in metric:
                            for threshold in [0.025, 0.05, 0.1, 0.2]:
                                score = early_exit_uq(
                                    metric_name=metric,  # type: ignore
                                    hidden_states=hidden_states,
                                    lm_head=lm_head,
                                    threshold=threshold,
                                )

                                score = sequence_ensemble(
                                    metric=score,
                                    last_layer_distribution=last_layer_distribution,
                                    hidden_states=hidden_states,
                                    attention_outputs=attention_outputs,
                                    attentions=attentions,
                                    pooling_ratio=pooling_ratio,
                                    weighting=weight_param,  # type: ignore
                                    prompt_length=prompt_length,
                                )

                                grid[metric][
                                    f"{agg}_{weight}_{threshold}"
                                ] = score.item()

                        elif "logits" in metric:
                            score = logits_uq(
                                metric_name=metric,  # type: ignore
                                hidden_states=hidden_states,
                                lm_head=lm_head,
                                sequences=sequences,
                            )

                            score = sequence_ensemble(
                                metric=score,
                                last_layer_distribution=last_layer_distribution,
                                hidden_states=hidden_states,
                                attention_outputs=attention_outputs,
                                attentions=attentions,
                                pooling_ratio=pooling_ratio,
                                weighting=weight_param,  # type: ignore
                                prompt_length=prompt_length,
                                quantile=0.75,
                            )

                            grid[metric][f"{agg}_{weight}"] = score.item()

                        elif "layer_evolution" in metric:
                            for layers_from_end in [3, 5, 10]:
                                score = layer_evolution_uq(
                                    metric_name=metric,  # type: ignore
                                    hidden_states=hidden_states,
                                    lm_head=lm_head,
                                    layers_from_end=layers_from_end,
                                )

                                score = sequence_ensemble(
                                    metric=score,
                                    last_layer_distribution=last_layer_distribution,
                                    hidden_states=hidden_states,
                                    attention_outputs=attention_outputs,
                                    attentions=attentions,
                                    pooling_ratio=pooling_ratio,
                                    weighting=weight_param,  # type: ignore
                                    prompt_length=prompt_length,
                                )

                                grid[metric][
                                    f"{agg}_{layers_from_end}_{weight}"
                                ] = score.item()

        completion += 1
        dataset_metrics.append(grid)

    # Compute metric AUROC
    results = {metric: {} for metric in metrics}
    for metric in metrics:
        for agg in aggregation_methods:
            for weight in weighting_methods:
                if "early_exit" in metric:
                    for threshold in [0.025, 0.05, 0.1, 0.2]:
                        key = f"{agg}_{weight}_{threshold}"
                        all_scores = []
                        all_failures = []

                        for item in dataset_metrics:
                            if key in item[metric]:
                                score = item[metric][key]
                                if score is not None and not torch.isnan(
                                    torch.tensor(score)
                                ):
                                    all_scores.append(item[metric][key])
                                    all_failures.append(not item["success"])

                        if len(set(all_failures)) > 1:
                            auroc = roc_auc_score(all_failures, all_scores)
                        else:
                            auroc = float("nan")

                        results[metric][key] = auroc

                elif "layer_evolution" in metric:
                    for layers_from_end in [3, 5, 10]:
                        key = f"{agg}_{layers_from_end}_{weight}"
                        all_scores = []
                        all_failures = []

                        for item in dataset_metrics:
                            if key in item[metric]:
                                score = item[metric][key]
                                if score is not None and not torch.isnan(
                                    torch.tensor(score)
                                ):
                                    all_scores.append(item[metric][key])
                                    all_failures.append(not item["success"])

                        if len(set(all_failures)) > 1:
                            auroc = roc_auc_score(all_failures, all_scores)
                        else:
                            auroc = float("nan")

                        results[metric][key] = auroc

                else:
                    key = f"{agg}_{weight}"
                    all_scores = []
                    all_failures = []

                    for item in dataset_metrics:
                        if key in item[metric]:
                            score = item[metric][key]
                            if score is not None and not torch.isnan(
                                torch.tensor(score)
                            ):
                                all_scores.append(item[metric][key])
                                all_failures.append(not item["success"])

                    if len(set(all_failures)) > 1:
                        auroc = roc_auc_score(all_failures, all_scores)
                    else:
                        auroc = float("nan")

                    results[metric][key] = auroc

    return results


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model name or path"
    )
    parser.add_argument(
        "--datafile", type=str, required=True, help="Path to the data file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the results",
    )

    parser.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        default=[],
        help="List of metrics to compute",
    )
    parser.add_argument(
        "--aggregation_methods",
        type=str,
        nargs="*",
        default=[],
        help="List of aggregation methods",
    )
    parser.add_argument(
        "--weighting_methods",
        type=str,
        nargs="*",
        default=[],
        help="List of weighting methods",
    )

    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, device_map="cpu"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = compute_uq_auroc_grid(
        datafile=args.datafile,
        model=model,
        tokenizer=tokenizer,
        metrics=args.metrics,
        aggregation_methods=args.aggregation_methods,
        weighting_methods=args.weighting_methods,
    )

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)
