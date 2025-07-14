from .gsm8k_prompt import system_prompt, user_prompt
from src.metrics.entropy import predictive_entropy, shannon_entropy
from src.utils.types import Metric, DatasetResult
from src.utils.inference import inference
from datasets import load_dataset


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
        model_answers=[],
        correct_answers=[],
        metrics={metric: [] for metric in metrics},
    )

    # Process in batches
    for i in range(0, len(indexes), batch_size):
        print(
            f"Processing batch {i // batch_size + 1} of {len(indexes) // batch_size + 1}"
        )

        batch_indexes = indexes[i : i + batch_size]

        # Prepare batch of messages
        batch_messages = []
        for idx in batch_indexes:
            question = gsm8k_ds[idx]["question"]  # type: ignore
            user_prompt_with_question = user_prompt.format(question=question)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_with_question},
            ]
            batch_messages.append(messages)

        # Run inference on batch
        output = inference(
            model=model,
            tokenizer=tokenizer,
            messages=batch_messages,
        )

        # Extract metrics for each item in batch
        if "predictive_entropy" in metrics:
            entropies = predictive_entropy(
                output.token_probabilities, output.sequence_length
            )
            results.metrics["predictive_entropy"].extend(entropies.tolist())

        if "shannon_entropy" in metrics:
            entropies = shannon_entropy(
                output.token_distribution, output.sequence_length
            )
            results.metrics["shannon_entropy"].extend(entropies.tolist())

        # Check answers for each item in batch
        for j, idx in enumerate(batch_indexes):
            model_answer = output.generated_text[j].split()[-1].strip(". ,$")
            correct_answer = gsm8k_ds[idx]["answer"].split()[-1].strip(". ,$")  # type: ignore

            results.incorrect_answers.append(correct_answer != model_answer)

            results.model_answers.append(output.generated_text[j])
            results.correct_answers.append(gsm8k_ds[idx]["answer"])  # type: ignore

    return results
