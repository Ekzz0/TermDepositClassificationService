from typing import Tuple
import pandas as pd
import joblib
from sklearn.metrics import recall_score, f1_score, precision_score
from .data_processing import split_to_x_y


def get_predict(test_data: pd.DataFrame) -> Tuple[float, float, float]:
    # Загрузка модели из файла
    filename = 'model.pkl'
    loaded_model = joblib.load(filename)

    # Подготовка к получению предтика
    X, y = split_to_x_y(test_data, 'y')

    predictions = loaded_model.predict(X.values)
    predictions_proba = loaded_model.predict_proba(X.values)

    result = pd.DataFrame(predictions_proba)
    result.index = test_data.index
    result.to_csv('result_example.csv', index=False)

    # Метрики
    recall = recall_score(y_true=y, y_pred=predictions, average='weighted')
    f1 = f1_score(y_true=y, y_pred=predictions, average='weighted')
    precision = precision_score(y_true=y, y_pred=predictions, average='weighted')
    return recall, f1, precision
