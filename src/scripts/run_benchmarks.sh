#!/bin/bash

export HF_ALLOW_CODE_EVAL="1"

tasks=(
  "mbpp"
  "gsm8k"
  "qasper"
  "squality"
  "humaneval"
)

model="microsoft/Phi-3.5-mini-instruct"
model_name="Phi-3.5-mini-instruct"

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
    lm_eval --model hf-store \
     --model_args pretrained="$model",dtype=float16,trust_remote_code=False \
     --apply_chat_template \
     --tasks "$item" \
     --confirm_run_unsafe_code \
     --device cuda:0 \
     --output_path src/data/evaluation_results/"$item"-"$model_name".json \
     --batch_size 1 \
     --log_samples \
     --limit 1

    # update sample metrics with the internal values from tensors
    python -m src.scripts.update_samples_metrics

    # delete all files in src/data/tensor_states
    rm -rf src/data/tensor_states/*
done
