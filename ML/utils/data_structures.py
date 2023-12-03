from dataclasses import dataclass

import pandas as pd


@dataclass
class ModelPrediction:
    precision: float
    recall: float
    f1: float
    predict: pd.DataFrame
