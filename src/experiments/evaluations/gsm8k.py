from .gsm8k_prompt import system_prompt, user_prompt
from src.metrics.entropy import predictive_entropy, shannon_entropy
from src.utils.types import Metric, DatasetResult
from src.utils.inference import inference
from datasets import load_dataset

# TODO: Implement batches and try it on a proper GPU


def gsm8k(
    model,
    tokenizer,
    metrics: list[Metric],
    indexes: list[int],
    batch_size: int = 1,
) -> DatasetResult:
    gsm8k_ds = load_dataset("openai/gsm8k", "main", split="train")

    results = DatasetResult(
        indexes=indexes,
        incorrect_answers=[],
        metrics={metric: [] for metric in metrics},
    )

    batch_size = min(batch_size, len(indexes))
    for i in indexes:
        question = gsm8k_ds[i]["question"]  # type: ignore
        user_prompt_with_question = user_prompt.format(question=question)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_with_question},
        ]

        output = inference(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
        )

        if "predictive_entropy" in metrics:
            results.metrics["predictive_entropy"].append(
                predictive_entropy(output.token_probabilities)[0].item()
            )

        if "shannon_entropy" in metrics:
            results.metrics["shannon_entropy"].append(
                shannon_entropy(output.token_distribution)[0].item()
            )

        model_answer = output.generated_text[0].split()[-1].strip(". ,$")
        correct_answer = gsm8k_ds[i]["answer"].split()[-1].strip(". ,$")  # type: ignore

        results.incorrect_answers.append(correct_answer != model_answer)

    return results
