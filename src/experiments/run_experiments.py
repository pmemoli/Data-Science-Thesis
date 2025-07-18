# Metric evaluations on gsm8k_ds

"""
1. Load model tokenizer and datasets
2. Define a list of elements of the dataset to evaluate
3. Evaluate the model performance on the dataset
4. Compute the different metrics
5. Compute the AUROC or whatever
"""

# %%
%reload_ext autoreload
%autoreload 2

from src.experiments.evaluations.gsm8k import gsm8k_evaluation
from src.utils.storage import store_results_as_csv
from src.utils.inference import inference
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from src.metrics.entropy import shannon_entropy, predictive_entropy

#%%
device = "cuda:0" 

# %%
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map=device,
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
gsm8k_ds = load_dataset("openai/gsm8k", "main", split="train")

# %%
messages = [
    [
        {
            "role": "system",
            "content": "You are a helpful assistant who answers concisely",
        },
        {
            "role": "user",
            "content": "Solve the following math problem: What is 2 + 2?",
        },
    ],
    [
        {
            "role": "system",
            "content": "You are a helpful assistant who answers concisely",
        },
        {
            "role": "user",
            "content": "Solve the following math problem: What is 2 + 5",
        },
    ],
]

# %%
output = inference(
    model=model,
    tokenizer=tokenizer,
    messages=messages,
    device=device,
)

# %%
indexes = [i for i in range(100)] 
results = gsm8k_evaluation(
    model=model,
    tokenizer=tokenizer,
    dataset=gsm8k_ds,
    metrics=["predictive_entropy", "shannon_entropy"],
    indexes=indexes,
    batch_size=5,
    device=device,
)

# %%
store_results_as_csv(
    result=results,
    csv_name="gsm8k_train_evaluation_results",
)
