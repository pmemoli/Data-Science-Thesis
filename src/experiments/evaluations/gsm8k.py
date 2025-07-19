from src.metrics.entropy import predictive_entropy, shannon_entropy
from src.utils.types import Metric, DatasetResult
from src.utils.inference import inference
import torch
import gc
import re


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
            on_hidden_states=False,
        )

        # Extract metrics for each item in batch
        entropies = None

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
            pattern = r"-?\d+(?:\.\d+)?"

            numbers = re.findall(pattern, output.generated_text[j])
            model_answer = numbers[-1] if numbers else "0"

            numbers = re.findall(pattern, dataset[idx]["answer"])
            correct_answer = numbers[-1] if numbers else "1"

            results.incorrect_answers.append(correct_answer != model_answer)

            results.model_answers.append(output.generated_text[j])
            results.correct_answers.append(dataset[idx]["answer"])  # type: ignore

        # Clear memory
        del output
        del entropies
        gc.collect()
        torch.cuda.empty_cache()

    return results


system_prompt = "You are a brilliant mathematician. Your task is to solve the following math problem by showing a step-by-step chain of thought. After your reasoning, state the final numerical answer clearly."

user_prompt = """Solve the following math problems. Show your reasoning step-by-step and then write the final answer in the format `#### <number>`.

---

Question: Natalia sold 48 liters of milk in the morning. In the afternoon, she sold 22 liters less than in the morning. How many liters of milk did she sell in total?

Answer: In the morning, Natalia sold 48 liters of milk. In the afternoon, she sold 22 liters less than in the morning, so she sold 48 - 22 = 26 liters. In total, she sold 48 + 26 = 74 liters.
#### 74

---

Question: A restaurant has 15 tables, and each table has 4 chairs. If 50 customers arrive, how many empty chairs will there be?

Answer: The restaurant has 15 tables * 4 chairs/table = 60 chairs in total. If 50 customers arrive, they will occupy 50 chairs. The number of empty chairs is 60 - 50 = 10.
#### 10

---

Question: John is reading a book with 300 pages. He reads 25 pages every day. After 7 days, how many pages are left for him to read?

Answer: John reads 25 pages/day * 7 days = 175 pages in 7 days. The book has 300 pages, so the number of pages left to read is 300 - 175 = 125.
#### 125

---

Question: A farmer collects 120 eggs. He packs them into cartons that hold a dozen eggs each. If he sells each carton for $3, how much money does he make?

Answer: A dozen is 12 eggs. The farmer has 120 eggs, so he can make 120 / 12 = 10 cartons. He sells each carton for $3, so he makes 10 cartons * $3/carton = $30.
#### 30

---

Question: The school library has 280 books. 3/7 of the books are fiction. How many non-fiction books are there?

Answer: The number of fiction books is (3/7) * 280. We can calculate this as (280 / 7) * 3 = 40 * 3 = 120 fiction books. The total number of books is 280, so the number of non-fiction books is 280 - 120 = 160.
#### 160

---

Question: A car travels at a speed of 60 km per hour. How many kilometers will it travel in 2 hours and 30 minutes?

Answer: 30 minutes is half an hour, or 0.5 hours. So, 2 hours and 30 minutes is 2.5 hours. The car travels at 60 km/h. The total distance traveled is 60 km/h * 2.5 hours = 150 km.
#### 150

---

Question: Lisa wants to buy a dress that costs $120. She has a coupon for a 20% discount. How much does she have to pay for the dress?

Answer: The discount is 20% of $120. We calculate the discount amount as 0.20 * 120 = $24. The final price is the original price minus the discount, which is $120 - $24 = $96.
#### 96

---

Question: A bakery uses 250 grams of flour for every loaf of bread. If they bake 15 loaves of bread, how many kilograms of flour do they use in total?

Answer: The bakery uses 250 grams of flour per loaf. For 15 loaves, they use 250 * 15 = 3750 grams of flour. The question asks for the answer in kilograms. Since 1 kilogram = 1000 grams, we convert grams to kilograms by dividing by 1000. So, 3750 grams / 1000 = 3.75 kilograms.
#### 3.75

---

Question: {question}

Answer: """
