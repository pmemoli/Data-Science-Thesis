import torch
import os
import hashlib

from lm_eval.models.huggingface import HFLM  # type: ignore

TENSOR_OUTPUT_DIR = "src/data/tensor_states"


class StateHFLM(HFLM):
    """
    A custom HFLM class from lm-evaluation-harness that stores the hidden states and attentions during evaluation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        os.makedirs(TENSOR_OUTPUT_DIR, exist_ok=True)

    def _get_prompt_hash(self, prompt):
        """Creates a unique and stable hash for a given prompt text."""
        return hashlib.md5(prompt.encode("utf-8")).hexdigest()

    def _save_states(self, unique_id, outputs):
        """Saves hidden states and attentions to a file."""
        tensor_path = os.path.join(TENSOR_OUTPUT_DIR, f"{unique_id}.pt")

        # Only stores generated token values. Detaches to cpu to avoid memory issues.
        to_save = {}
        if (
            hasattr(outputs, "hidden_states")
            and outputs.hidden_states is not None
        ):
            to_save["hidden_states"] = [
                h.detach().cpu() for h in outputs.hidden_states[0]
            ]

        if hasattr(outputs, "attentions") and outputs.attentions is not None:
            to_save["attentions"] = [
                a.detach().cpu() for a in outputs.attentions[0]
            ]

        if to_save:
            torch.save(to_save, tensor_path)

    def generate_until(self, requests, **kwargs):
        """Generates text and captures hidden states and attentions."""

        results = super().generate_until(requests, **kwargs)

        # Re-run inference to capture the states. Very unoptimized but works for now.
        # gen_kwargs = kwargs.get("generation_kwargs", {})
        # gen_kwargs["output_hidden_states"] = True
        # gen_kwargs["output_attentions"] = True
        # gen_kwargs["return_dict_in_generate"] = True
        # kwargs["generation_kwargs"] = gen_kwargs
        #
        # for _, req in enumerate(requests):
        #     prompt = req.args[0]
        #     unique_id = self._get_prompt_hash(prompt)
        #
        #     with torch.no_grad():
        #         tokenized_input = self.tokenizer(
        #             prompt, return_tensors="pt"
        #         ).to(self.device)
        #         outputs = self.model.generate(
        #             input_ids=tokenized_input.input_ids,
        #             attention_mask=tokenized_input.attention_mask,
        #             **kwargs["generation_kwargs"],
        #         )
        #     self._save_states(unique_id, outputs)

        return results

    def loglikelihood(self, requests, **kwargs):
        """Computes log-likelihood and captures hidden states and attentions."""

        results = super().loglikelihood(requests, **kwargs)

        # Re-run inference to capture the states. Very unoptimized but works for now.
        # gen_kwargs = kwargs.get("generation_kwargs", {})
        # gen_kwargs["output_hidden_states"] = True
        # gen_kwargs["output_attentions"] = True
        # gen_kwargs["return_dict_in_generate"] = True
        # kwargs["generation_kwargs"] = gen_kwargs
        #
        # with torch.no_grad():
        #     for _, req in enumerate(requests):
        #         prompt, context = req.args
        #         unique_id = self._get_prompt_hash(prompt)
        #
        #         tokenized_input = self.tokenizer(
        #             context, return_tensors="pt"
        #         ).to(self.device)
        #         outputs = self.model(
        #             input_ids=tokenized_input.input_ids,
        #             attention_mask=tokenized_input.attention_mask,
        #             output_hidden_states=True,
        #             output_attentions=True,
        #         )
        #         self._save_states(unique_id, outputs)

        return results
