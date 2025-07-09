from transformers import pipeline, PreTrainedModel, PreTrainedTokenizer
import torch


# I should be returning an object with the generated text and token probabilitiesfrom which I compute the metrics.


def inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    system_prompt: str,
    user_prompt: str,
    seed: int = 42,
) -> str:

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    generator = torch.Generator().manual_seed(seed)
    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 1.0,
        "do_sample": True,
        "generator": generator,
    }
    output = pipe(messages, **generation_args)

    return output[0]["generated_text"]  # type: ignore
