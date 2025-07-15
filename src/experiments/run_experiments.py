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
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# %%
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
gsm8k_ds = load_dataset("openai/gsm8k", "main", split="train")

# %%
indexes = [i for i in range(4)] 
results = gsm8k_evaluation(
    model=model,
    tokenizer=tokenizer,
    dataset=gsm8k_ds,
    metrics=["predictive_entropy", "shannon_entropy"],
    indexes=indexes,
    batch_size=2,
)

# %%
store_results_as_csv(
    result=results,
    csv_name="gsm8k_train_evaluation_results",
)
