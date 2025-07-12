# Metric evaluations on gsm8k_ds

"""
1. Load model tokenizer and datasets
2. Define a list of elements of the dataset to evaluate
3. Evaluate the model performance on the dataset
4. Compute the different metrics
5. Compute the AUROC or whatever
"""

# %%
from src.experiments.math_prompt import system_prompt, user_prompt
from src.utils.inference import inference
from src.metrics.entropy import predictive_entropy, shannon_entropy
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
problems = gsm8k_ds[:3]["question"]  # type: ignore
answers = gsm8k_ds[:3]["answer"]  # type: ignore

# %% Compute the model's answer and metrics for gsm8k
incorrect_answers = []

predictive_entropy_metric = []
shannon_entropy_metric = []
for i in range(len(problems)):
    question = problems[i]
    user_prompt_with_question = user_prompt.format(question=question)

    print(f"Question: {question} \n\n")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_with_question},
    ]

    output = inference(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
    )

    predictive_entropy_metric.append(
        predictive_entropy(output.token_probabilities)
    )
    shannon_entropy_metric.append(shannon_entropy(output.token_distribution))

    model_answer = output.generated_text[0].split()[-1].strip(". ,$")
    correct_answer = answers[i].split()[-1].strip(". ,$")

    incorrect_answers.append(correct_answer != model_answer)

    print(f"Model's answer: {model_answer}")
    print(f"Correct answer: {correct_answer}")

# %%
print("\n\n")
print(f"Predictive entropy metric: {predictive_entropy_metric}")
print(f"Shannon entropy metric: {shannon_entropy_metric}")
print(f"Incorrect answers: {incorrect_answers}")
