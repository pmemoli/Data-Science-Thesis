from dataclasses import dataclass
from typing import Literal
from torch import Tensor

Metric = Literal["predictive_entropy", "shannon_entropy"]


@dataclass
class DatasetResult:
    indexes: list[int]
    incorrect_answers: list[bool]
    metrics: dict[Metric, list[float]]
    model_answers: list[str]
    correct_answers: list[str]


@dataclass
class InferenceOutput:
    # [batch_size][sequence_length]
    generated_ids: Tensor
    sequence_length: Tensor
    token_probabilities: Tensor

    # [batch_size][layer_amount][sequence_length][top_k]
    token_distribution: Tensor
    token_distribution_ids: Tensor

    # [batch_size]
    generated_text: list[str]
