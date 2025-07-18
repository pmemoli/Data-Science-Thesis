from src.metrics.entropy import predictive_entropy, shannon_entropy
from src.utils.types import Metric, DatasetResult
from src.utils.inference import inference


def gsm8k_evaluation(
    model,
    tokenizer,
    dataset,
    metrics: list[Metric],
    indexes: list[int],
    batch_size: int = 1,
    device: str = "auto",
) -> DatasetResult:
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
            question = dataset[idx]["question"]  # type: ignore
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
            device=device,
        )

        # Extract metrics for each item in batch
        if "predictive_entropy" in metrics:
            entropies = predictive_entropy(
                output.token_probabilities,
                output.sequence_length,
                device=device,
            )
            results.metrics["predictive_entropy"].extend(entropies.tolist())

        if "shannon_entropy" in metrics:
            entropies = shannon_entropy(
                output.token_distribution,
                output.sequence_length,
                layer=-1,
                device=device,
            )
            results.metrics["shannon_entropy"].extend(entropies.tolist())

        # Check answers for each item in batch
        for j, idx in enumerate(batch_indexes):
            model_answer = output.generated_text[j].split()[-1].strip(". ,$")
            correct_answer = dataset[idx]["answer"].split()[-1].strip(". ,$")  # type: ignore

            results.incorrect_answers.append(correct_answer != model_answer)

            results.model_answers.append(output.generated_text[j])
            results.correct_answers.append(dataset[idx]["answer"])  # type: ignore

    return results


system_prompt = "You are a brilliant mathematician. Your task is to solve the following math problem by showing a step-by-step chain of thought. After your reasoning, state the final numerical answer clearly."

user_prompt = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for $3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.

Q: {question}
A: """
