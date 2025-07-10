from transformers import PreTrainedModel, PreTrainedTokenizer

# from src.utils import inference


def semantic_entropy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    system_prompt: str,
    user_prompt: str,
    seed: int = 42,
) -> float:

    return 0
