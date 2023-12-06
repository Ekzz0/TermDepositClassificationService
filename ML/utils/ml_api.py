from .feature_constructor import FeatureConstructor
from .loaded_model import LoadedModel


def load_model(path: str) -> LoadedModel:
    return LoadedModel(path)


def load_feature_constructor() -> FeatureConstructor:
    return FeatureConstructor()
