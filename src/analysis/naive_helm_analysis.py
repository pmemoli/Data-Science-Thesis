# %%
%reload_ext autoreload
%autoreload 2

from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
import json

# %%
model = "microsoft_phi-3.5-mini-instruct"
suite = "helm-lite-cot-zeroshot"
data_path = f"src/data/helm/runs/{suite}"

by_scenario = {
    "narrative_qa": [f"narrative_qa:model={model},max_train_instances=0"],
    "mmlu": [
        f"mmlu:subject=abstract_algebra,method=multiple_choice_joint,model={model},max_train_instances=0",
        f"mmlu:subject=college_chemistry,method=multiple_choice_joint,model={model},max_train_instances=0",
        f"mmlu:subject=computer_security,method=multiple_choice_joint,model={model},max_train_instances=0",
        f"mmlu:subject=econometrics,method=multiple_choice_joint,model={model},max_train_instances=0",
        f"mmlu:subject=us_foreign_policy,method=multiple_choice_joint,model={model},max_train_instances=0",
    ],
    "math": [
        f"math:subject=number_theory,level=1,use_official_examples=False,use_chain_of_thought=True,model={model},max_train_instances=0",
        f"math:subject=intermediate_algebra,level=1,use_official_examples=False,use_chain_of_thought=True,model={model},max_train_instances=0",
        f"math:subject=algebra,level=1,use_official_examples=False,use_chain_of_thought=True,model={model},max_train_instances=0",
        f"math:subject=prealgebra,level=1,use_official_examples=False,use_chain_of_thought=True,model={model},max_train_instances=0",
        f"math:subject=geometry,level=1,use_official_examples=False,use_chain_of_thought=True,model={model},max_train_instances=0",
        f"math:subject=counting_and_probability,level=1,use_official_examples=False,use_chain_of_thought=True,model={model},max_train_instances=0",
        f"math:subject=precalculus,level=1,use_official_examples=False,use_chain_of_thought=True,model={model},max_train_instances=0",
    ],
    "gsm": [f"gsm:model={model},stop=none,max_train_instances=0"],
    "legalbench": [
        f"legalbench:subset=abercrombie,model={model},max_train_instances=0",
        f"legalbench:subset=corporate_lobbying,model={model},max_train_instances=0",
        f"legalbench:subset=international_citizenship_questions,model={model},max_train_instances=0",
        f"legalbench:subset=function_of_decision_section,model={model},max_train_instances=0",
        f"legalbench:subset=proa,model={model},max_train_instances=0",
    ],
    "med_qa": [f"med_qa:model={model},max_train_instances=0"],
}

by_domain = {
    "mathematical_reasoning": ["math", "gsm"],
    "knowledge_qa": ["mmlu", "natural_qa", "openbookqa", "narrative_qa", "commonsense"],
    "specialized_domains": ["med_qa", "legalbench"],
    "language_tasks": ["wmt_14"]
}

# %%
columns = [
    "scenario",
    "run",
    "domain",
    "benchmark",
    "sequence_negative_log_likelihood",
    "auroc_sequence_negative_log_likelihood",
    "max_token_negative_log_likelihood",
    "auroc_max_token_negative_log_likelihood",
    "predictive_entropy",
    "auroc_predictive_entropy",
    "shannon_entropy",
    "auroc_shannon_entropy",
    "model"
]

values = []

for scenario, runs in by_scenario.items():
    domain = "unknown"
    for d, scenarios in by_domain.items():
        if scenario in scenarios:
            domain = d
            break

    if scenario == "math":
        continue

    for run in runs: 
        path = f"{data_path}/{run}/scenario_state.json"
        with open(path, "r") as f:
            scenario_state = json.load(f)

        instances = scenario_state["request_states"]

        average_metrics = {
            "sequence_negative_log_likelihood": 0, 
            "max_token_negative_log_likelihood": 0, 
            "predictive_entropy": 0, 
            "shannon_entropy": 0
        }

        contains_errors = []

        amount_of_success = 0
        for instance in instances:
            metrics = instance["result"]["completions"][0]["metrics"]
            evaluation = instance["evaluation"]
            contains_errors.append(not evaluation)

            if evaluation:
                amount_of_success += 1

            for key in average_metrics.keys():
                average_metrics[key] += metrics[key] / len(instances)

        benchmark_score = amount_of_success / len(instances)

        nll_auroc = roc_auc_score(contains_errors, [instance["result"]["completions"][0]["metrics"]["sequence_negative_log_likelihood"] for instance in instances])

        max_nll_auroc = roc_auc_score(contains_errors, [instance["result"]["completions"][0]["metrics"]["max_token_negative_log_likelihood"] for instance in instances])

        predictive_entropy_auroc = roc_auc_score(contains_errors, [instance["result"]["completions"][0]["metrics"]["predictive_entropy"] for instance in instances])

        shannon_entropy_auroc = roc_auc_score(contains_errors, [instance["result"]["completions"][0]["metrics"]["shannon_entropy"] for instance in instances])


        values.append([
            scenario,
            run,
            domain,
            benchmark_score,
            average_metrics["sequence_negative_log_likelihood"],
            nll_auroc,
            average_metrics["max_token_negative_log_likelihood"],
            max_nll_auroc,
            average_metrics["predictive_entropy"],
            predictive_entropy_auroc,
            average_metrics["shannon_entropy"],
            shannon_entropy_auroc,
            model
        ])

for scenario, runs in by_scenario.items():
    if scenario != "math":
        continue

    average_metrics = {
        "sequence_negative_log_likelihood": 0, 
        "max_token_negative_log_likelihood": 0, 
        "predictive_entropy": 0, 
        "shannon_entropy": 0
    }

    contains_errors = []
    instances = []

    amount_of_items = 0

    for run in runs: 
        path = f"{data_path}/{run}/scenario_state.json"
        with open(path, "r") as f:
            scenario_state = json.load(f)

        instances.extend(scenario_state["request_states"])

    amount_of_success = 0
    for instance in instances:
        metrics = instance["result"]["completions"][0]["metrics"]
        evaluation = instance["evaluation"]
        contains_errors.append(not evaluation)

        if evaluation:
            amount_of_success += 1

        for key in average_metrics.keys():
            average_metrics[key] += metrics[key] / len(instances)

    benchmark_score = amount_of_success / len(instances)

    nll_auroc = roc_auc_score(contains_errors, [instance["result"]["completions"][0]["metrics"]["sequence_negative_log_likelihood"] for instance in instances])

    max_nll_auroc = roc_auc_score(contains_errors, [instance["result"]["completions"][0]["metrics"]["max_token_negative_log_likelihood"] for instance in instances])

    predictive_entropy_auroc = roc_auc_score(contains_errors, [instance["result"]["completions"][0]["metrics"]["predictive_entropy"] for instance in instances])

    shannon_entropy_auroc = roc_auc_score(contains_errors, [instance["result"]["completions"][0]["metrics"]["shannon_entropy"] for instance in instances])


    values.append([
        scenario,
        "math",
        "mathematical_reasoning",
        benchmark_score,
        average_metrics["sequence_negative_log_likelihood"],
        nll_auroc,
        average_metrics["max_token_negative_log_likelihood"],
        max_nll_auroc,
        average_metrics["predictive_entropy"],
        predictive_entropy_auroc,
        average_metrics["shannon_entropy"],
        shannon_entropy_auroc,
        model
    ])


# %%
df = pd.DataFrame(values, columns=columns) # type: ignore
df['domain'] = pd.Categorical(df['domain'], categories=by_domain.keys(), ordered=True)
df = df.sort_values(['domain', 'scenario', 'run'])
df = df.reset_index(drop=True)

#%% save df as csv
df.to_csv(f"src/data/tables/cot_helm_analysis.csv", index=False)

