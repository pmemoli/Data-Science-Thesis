from src.utils.types import InferenceOutput
import torch.nn.functional as F
import torch


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
        output_hidden_states=True,
    )

    # Generated sequence text and ids
    sequences = outputs.sequences
    prompt_length = inputs.shape[1]

    generated_ids = sequences[:, prompt_length:]
    generated_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True
    )

    # Token distribution (for all hidden states)
    generatation_probabilities = []
    for step, layer_outputs in enumerate(outputs.hidden_states):
        step_probabilities = []

        for layer in range(len(outputs.hidden_states[step])):
            if step != 0:
                layer_hidden = layer_outputs[layer].squeeze(1)
            else:
                layer_hidden = layer_outputs[layer][:, -1, :]

            logits = model.lm_head(layer_hidden)
            probabilities = F.softmax(logits, dim=-1)

            step_probabilities.append(probabilities)

        step_tensor = torch.stack(step_probabilities, dim=1)
        generatation_probabilities.append(step_tensor)

    # [batch_size, layers, sequence_length, vocab_size]
    token_distribution = torch.stack(generatation_probabilities, dim=2)

    # Generated sequence lengths
    eos_id = tokenizer.eos_token_id
    generated_tokens = torch.where(generated_ids != eos_id, 1, 0)
    sequence_length = generated_tokens.sum(dim=-1)

    return InferenceOutput(
        generated_ids=generated_ids,
        sequence_length=sequence_length,
        generated_text=generated_text,
        token_distribution=token_distribution,
    )
