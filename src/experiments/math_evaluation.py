# Metric evaluations on math datasets

"""
1. Load model tokenizer and datasets
2. Define a list of elements of the dataset to evaluate
3. Evaluate the model performance on the dataset
4. Compute the different metrics
5. Compute the AUROC or whatever
"""

# %%
from src.utils.inference import inference
from datasets import load_dataset

# %%
math_ds = load_dataset("nlile/hendrycks-MATH-benchmark", split="train")
gsm8k_ds = load_dataset("openai/gsm8k", "main", split="train")

# %%
system_prompt = "Be concise and answer the question directly. Do not provide any additional information."

# On monday or tuesday get the auroc for both datasets on the 2 entropy metrics
