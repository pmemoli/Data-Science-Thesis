from dataclasses import dataclass
import torch.nn.functional as F
import torch


@dataclass
class InferenceOutput:
    # [batch_size]
    generated_text: list[str]

    # [batch_size][sequence_length][top_k]
    token_probabilities: list[list[tuple[str, float]]]


def inference(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    seed: int = 42,
) -> InferenceOutput:
    torch.manual_seed(seed)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").input_ids

    # Inference
    top_k = 50
    outputs = model.generate(
        inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=1.0,
        top_k=top_k,
        top_p=0.98,
        return_dict_in_generate=True,
        output_scores=True,
    )

    # Sequence probabilities
    scores = outputs.scores
    logits = torch.stack(scores, dim=1)

    # Get the top k tokens with their probabilities
    probabilities = F.softmax(logits, dim=-1)
    top_k_probabilities, top_k_indices = torch.topk(probabilities, top_k)

    # Convert indices to tokens and sort them by probability
    top_k_probabilities_shape = top_k_probabilities.size()

    batch_token_probs = []  # [batch_size][sequence_length][top_k]
    for i in range(top_k_probabilities_shape[0]):
        sequence_token_probs = []
        for j in range(top_k_probabilities_shape[1]):
            token_probs = []
            for k in range(top_k_probabilities_shape[2]):
                probability = top_k_probabilities[i][j][k].item()
                token_index = top_k_indices[i][j][k].item()
                token = tokenizer.decode(token_index)

                token_probs.append((token, probability))
            sequence_token_probs.append(token_probs)
        batch_token_probs.append(sequence_token_probs)

    # Generated sequence
    sequences = outputs.sequences
    generated_text = tokenizer.batch_decode(sequences)

    return InferenceOutput(
        generated_text=generated_text,
        token_probabilities=batch_token_probs,
    )
