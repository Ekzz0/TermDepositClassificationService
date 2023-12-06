from dataclasses import dataclass

@dataclass
class Person:
    probability: float
    ID: int


@dataclass
class PersonsList:
    persons: list[Person]


@dataclass
class ModelPrediction:
    result: PersonsList


@dataclass
class Score:
    precision: float
    recall: float
    f1: float


@dataclass
class FeatureImportance:
    features: list[str]
    importance: list[float]


