from dataclasses import dataclass
from pydantic import BaseModel


# @dataclass
# class Person:
#     probability: float
#     ID: int
#
#
# @dataclass
# class PersonsList:
#     persons: list[Person]


# @dataclass
# class ModelPrediction:
#     result: PersonsList

# Класс для response после fit
class Score(BaseModel):
    precision: float
    recall: float
    f1: float


@dataclass
class FeatureImportance:
    features: list[str]
    importance: list[float]


