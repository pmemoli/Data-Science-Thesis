from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.scenarios import REGISTRY
from src.evaluation.core.utils import (
    hidden_states_reshape,
    attentions_reshape,
    attention_outputs_reshape,
)
from src.metrics.token_uq import logits_uq

import argparse
import torch
import time
import gc
import os


def run_benchmark(
    dataset_name: str,
    model_name: str,
    suite: str,
    result_path: str,
    temperature: float = 0.5,
    max_length: int = 1024,
    store_tensors: bool = False,
    store_metrics: bool = False,
    store_logprobs: bool = False,
    device: str = "cuda:0",
    limit: int | None = None,
):
    ScenarioClass = REGISTRY[dataset_name]
    scenario = ScenarioClass()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
        trust_remote_code=False,
        device_map="cuda:0",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=False,
    )

    # Add hook to store attention outputs
    attention_outputs = {}

    def save_attention_output(layer_idx):
        def hook(module, input, output):
            # Store attention output for this generation step
            if f"layer_{layer_idx}" not in attention_outputs:
                attention_outputs[f"layer_{layer_idx}"] = []
            attention_outputs[f"layer_{layer_idx}"].append(
                output[0].detach().cpu()
            )

        return hook

    for i, layer in enumerate(model.model.layers):
        layer.self_attn.register_forward_hook(save_attention_output(i))

    # Processes one item at a time for simplicity
    results = []
    amount_processed = 0
    while scenario.has_next() and (limit is None or amount_processed < limit):
        print(f"Processing sample {amount_processed + 1}...")

        sample = scenario.sample()
        if sample is None:
            break

        prompt = sample["prompt"]
        reference = sample["reference"]

        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,  # Adds proper assistant prompt
            return_tensors="pt",
            return_dict=True,  # Returns dict with input_ids AND attention_mask
        ).to(device=device)

        prompt_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_length,
                temperature=temperature,
                num_return_sequences=1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_attentions=True,
            )

        output_sequence = outputs.sequences[0, prompt_length:]
        decoded_sequence = tokenizer.decode(
            output_sequence,
            skip_special_tokens=True,
        )

        result = {
            "prompt": prompt,
            "prompt_length": prompt_length,
            "reference": reference,
            "generation": decoded_sequence,
        }

        if store_tensors or store_metrics or store_logprobs:
            hidden_states = hidden_states_reshape(outputs.hidden_states)
            full_sequence = outputs.sequences[0]

            if store_tensors:
                attentions = attentions_reshape(outputs.attentions)
                attentions_outputs = attention_outputs_reshape(
                    attention_outputs
                )

                result["hidden_states"] = hidden_states.cpu()
                result["attentions"] = attentions.cpu()
                result["attention_outputs"] = attentions_outputs.cpu()
                result["sequences"] = full_sequence.cpu()

            if store_logprobs:
                shannon_uq = logits_uq(
                    hidden_states,
                    model.lm_head,
                    full_sequence,
                    "logits_shannon_entropy",
                )

                result["token_shannon_entropy"] = shannon_uq.cpu()

        gc.collect()
        attention_outputs.clear()
        attention_outputs = {}

        results.append(result)
        amount_processed += 1

        # Progress indicator
        if amount_processed % 10 == 0:
            print(f"Processed {amount_processed} samples...")

    # Store results
    file_path = f"{result_path}/{suite}"
    os.makedirs(file_path, exist_ok=True)

    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = f"{file_path}/{dataset_name}_{model_name.replace('/', '_')}_{time_stamp}.pt"
    torch.save(results, output_file)

    print(f"Benchmark completed. Results saved to: {output_file}")
    print(f"Total samples processed: {amount_processed}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation on language models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset to use (must be registered in REGISTRY)",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name or path of the model to evaluate",
    )

    parser.add_argument(
        "--suite", type=str, required=True, help="Name of the evaluation suite"
    )

    parser.add_argument(
        "--result_path", type=str, required=True, help="Path to save results"
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature for text generation",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum length for generated text",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run the model on (e.g., 'cuda:0', 'cpu')",
    )

    parser.add_argument(
        "--store_tensors",
        type=bool,
        required=True,
        help="Whether to store hidden states and attentions (True/False)",
    )

    parser.add_argument(
        "--store_metrics",
        type=bool,
        default=False,
        help="Whether to store evaluation metrics (True/False)",
    )

    parser.add_argument(
        "--store_logprobs",
        type=bool,
        default=False,
        help="Whether to store token log probabilities [fixed to shannon entropy] (True/False)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of samples to process (useful for testing)",
    )

    return parser.parse_args()


def main():
    """Main entry point for command line execution."""
    try:
        args = parse_arguments()

        # Validate dataset name
        if args.dataset_name not in REGISTRY:
            available_datasets = list(REGISTRY.keys())
            raise ValueError(
                f"Dataset '{args.dataset_name}' not found in registry. "
                f"Available datasets: {available_datasets}"
            )

        # Validate device
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            print(
                "Warning: CUDA requested but not available. Falling back to CPU."
            )
            args.device = "cpu"

        print("\nStarting benchmark with the following configuration:\n")
        print(f"  Dataset: {args.dataset_name}")
        print(f"  Model: {args.model_name}")
        print(f"  Suite: {args.suite}")
        print(f"  Device: {args.device}")
        print(f"  Temperature: {args.temperature}")
        print(f"  Max length: {args.max_length}")
        print(f"  Limit: {args.limit if args.limit else 'No limit'}")
        print(f"  Results will be saved to: {args.result_path}")
        print("-" * 30)

        run_benchmark(
            dataset_name=args.dataset_name,
            model_name=args.model_name,
            suite=args.suite,
            result_path=args.result_path,
            temperature=args.temperature,
            max_length=args.max_length,
            device=args.device,
            limit=args.limit,
            store_tensors=args.store_tensors,
            store_metrics=args.store_metrics,
            store_logprobs=args.store_logprobs,
        )

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"Error running benchmark: {str(e)}")
        raise


if __name__ == "__main__":
    main()
