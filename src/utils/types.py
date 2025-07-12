from dataclasses import dataclass
from torch import Tensor


@dataclass
class InferenceOutput:
    # [batch_size]
    generated_text: list[str]

    # [batch_size][sequence_length][top_k]
    token_distribution: Tensor

    # [batch_size][sequence_length]
    token_probabilities: Tensor
