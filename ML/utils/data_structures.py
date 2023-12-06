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
    precision: float
    recall: float
    f1: float
    result: PersonsList


@dataclass
class FeatureImportance:
    features: list[str]
    importance: list[float]
