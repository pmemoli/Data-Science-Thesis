from src.utils.types import DatasetResult
import pandas as pd


def store_results_as_csv(result: DatasetResult, csv_name: str) -> None:
    data_for_df = {
        "index": result.indexes,
        "is_incorrect": result.incorrect_answers,
        "model_answer": result.model_answers,
        "correct_answer": result.correct_answers,
    }
    data_for_df.update(result.metrics)  # type: ignore

    df = pd.DataFrame(data_for_df)
    path = "src/experiments/results"
    df.to_csv(f"{path}/{csv_name}.csv", index=False)
