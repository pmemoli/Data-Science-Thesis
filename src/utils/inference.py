import torch.nn.functional as F
from dataclasses import dataclass
import torch


@dataclass
class InferenceOutput:
    # [batch_size]
    generated_ids: torch.Tensor
    sequence_length: torch.Tensor
    generated_text: list[str]

    # [batch_size][sequence_length][top_k]
    token_distribution: torch.Tensor

    # [batch_size][sequence_length]
    token_probabilities: torch.Tensor


def inference(
    model,
    tokenizer,
    messages: list[list[dict]],
    seed: int = 42,
) -> InferenceOutput:
    torch.manual_seed(seed)

    # Input
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt", padding=True).input_ids

    # Output generation
    top_k = 50
    outputs = model.generate(
        inputs,
        max_new_tokens=1000,
        do_sample=True,
        temperature=1.0,
        top_k=top_k,
        top_p=0.98,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Generated sequence text and ids
    sequences = outputs.sequences
    prompt_length = inputs.shape[1]

    generated_ids = sequences[:, prompt_length:]
    generated_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True
    )

    # Token distribution
    scores = outputs.scores
    logits = torch.stack(scores, dim=1)
    probabilities = F.softmax(logits, dim=-1)

    # Selected token probabilities
    indices = generated_ids.unsqueeze(-1)
    token_probabilities = torch.gather(probabilities, 2, indices).squeeze(-1)

    # Generated sequence lengths
    eos_id = tokenizer.eos_token_id
    generated_tokens = torch.where(generated_ids != eos_id, 1, 0)
    sequence_length = generated_tokens.sum(dim=-1)

    return InferenceOutput(
        generated_ids=generated_ids,
        sequence_length=sequence_length,
        generated_text=generated_text,
        token_distribution=probabilities,
        token_probabilities=token_probabilities,
    )
