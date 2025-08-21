#!/bin/bash

export HF_ALLOW_CODE_EVAL="1"

tasks=(
  "arena-hard"
  "bbh"
  "mmlu_college_math"
  "mmlu_hs_math"
  "mmlu_pro"
  "arc_challenge"
  "boolq"
  "gpqa"
  "hellaswag"
  "openbookqa"
  "piqa"
  "social_i_qa"
  "truthfulqa_mc2"
  "winogrande"
  "mmlu_multilingual"
  "mgsm"
  "gsm8k"
  "math"
  "qasper"
  "squality"
  "humaneval"
  "mbpp"
)

declare -A bench_signature

model="microsoft/phi-3.5-mini-instruct"
model_name="phi-3.5-mini-instruct"

# Prompt user for task
read -p "Enter a task to run (press Enter to run all tasks): " input_task

# Determine which tasks to run
if [ -z "$input_task" ]; then
    selected_tasks=("${tasks[@]}")  # run all tasks
else
    selected_tasks=("$input_task")  # run only the task user entered
fi

for item in "${selected_tasks[@]}"; do
    echo "Running evaluation for task: $item with model: $model_name"

    # run lm_eval with the specified model and task
    helm-run --run-entries gsm:model=microsoft/phi-3.5-mini-instruct \
     --suite naive-eval \
     --output-path src/data/helm/ \
     --max-eval-instances 100
done
