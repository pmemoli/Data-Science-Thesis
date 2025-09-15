from transformers import AutoModelForCausalLM, AutoTokenizer

from src.engine.scenarios import REGISTRY
from src.engine.core.utils import hidden_states_reshape, attentions_reshape

import argparse
import torch
import time
import json
import os


def run_benchmark(
    dataset_name: str,
    model_name: str,
    suite: str,
    result_path: str,
    temperature: float = 0.5,
    max_length: int = 1024,
    batch_size: int = 1,
    sample_amount: int = 1,
    store_tensors: bool = False,
    store_metrics: bool = False,
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

        hidden_states = hidden_states_reshape(outputs.hidden_states).to(
            device=device
        )
        attentions = attentions_reshape(outputs.attentions, prompt_length).to(
            device=device
        )
        sequences = outputs.sequences[:, prompt_length:].to(device=device)
        string_sequence = tokenizer.decode(
            sequences[0],
            skip_special_tokens=True,
        )

        result = {
            "prompt": prompt,
            "prompt_length": prompt_length,
            "reference": reference,
            "generation": string_sequence,
            "hidden_states": hidden_states.cpu() if store_tensors else None,
            "attentions": attentions.cpu() if store_tensors else None,
            "sequences": sequences.cpu() if store_tensors else None,
        }

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
        "--batch_size", type=int, default=1, help="Batch size for processing"
    )

    parser.add_argument(
        "--sample_amount",
        type=int,
        default=1,
        help="Number of samples to generate per prompt",
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
            batch_size=args.batch_size,
            sample_amount=args.sample_amount,
            device=args.device,
            limit=args.limit,
            store_tensors=args.store_tensors,
            store_metrics=args.store_metrics,
        )

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"Error running benchmark: {str(e)}")
        raise


if __name__ == "__main__":
    main()
