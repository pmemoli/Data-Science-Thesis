from dataclasses import dataclass
from typing import Literal

Metric = Literal["predictive_entropy", "shannon_entropy"]


@dataclass
class DatasetResult:
    indexes: list[int]
    incorrect_answers: list[bool]
    metrics: dict[Metric, list[float]]
    model_answers: list[str]
    correct_answers: list[str]
