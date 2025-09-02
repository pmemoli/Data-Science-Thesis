from .scenarios import REGISTRY
from transformers import pipeline


def run_benchmark(
    dataset_name: str,
    model_name: str,
    suite: str,
    result_path: str,
    temperature:float = 0.5,
    max_length:int = 1024,
    batch_size:int = 1,
    sample_amount:int = 1,
    metrics: list[str] = [],
    validate_with_external_llm: bool = False,
    limit:int | None = None,
):
    ScenarioClass = REGISTRY[dataset_name]
    scenario = ScenarioClass()
    pipe = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=model_name,
        trust_remote_code=False,
    )
    
    # Processes one item at a time for simplicity
    amount_processed = 0
    while scenario.has_next() or (limit and amount_processed < limit):
        sample = scenario.sample()
        if sample is None:
            break

        prompt = sample["prompt"]
        reference = sample["reference"]

        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ]

        outputs = pipe(
            messages,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=1,
            return_full_text=False,
            truncation=True,
            pad_token_id=pipe.tokenizer.eos_token_id,
            eos_token_id=pipe.tokenizer.eos_token_id,
            do_sample=True,
        )

        for output in outputs:
            generated_text = output["generated_text"]

            print("Prompt:", prompt)
            print("\nReference:", reference)
            print("\nGenerated:", generated_text)

        # compute metrics

        amount_processed += 1
