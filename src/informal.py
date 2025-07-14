# %%
%reload_ext autoreload
%autoreload 2

import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.experiments.evaluations.gsm8k import gsm8k
from src.utils.inference import inference
from src.metrics.entropy import predictive_entropy, shannon_entropy
from datasets import load_dataset
import torch

# %%
math_ds = load_dataset("nlile/hendrycks-MATH-benchmark", split="train")
gsm8k_ds = load_dataset("openai/gsm8k", "main", split="train")

# %%
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

#%%
test_results = gsm8k(
    model=model,
    tokenizer=tokenizer,
    metrics=["predictive_entropy", "shannon_entropy"],
    batch_size=4,
    indexes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
)

#%%
print(test_results.incorrect_answers)
print(test_results.metrics["predictive_entropy"])
