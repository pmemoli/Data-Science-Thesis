from lm_eval.api.registry import register_model  # type: ignore
from src.models.hflm_eval import StateHFLM

register_model("state_hf", StateHFLM)
