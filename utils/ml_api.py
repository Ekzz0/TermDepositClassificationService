from typing import Tuple
import pandas as pd
import joblib
from sklearn.metrics import recall_score, f1_score, precision_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from .data_processing import split_to_x_y


def get_predict(df: pd.DataFrame, filename: str = 'RandomForest') -> Tuple[float, float, float]:
    # Загрузка модели из файла
    loaded_model = joblib.load(filename + '.pkl')

    # Подготовка к получению предтика
    X, y = split_to_x_y(df, 'y')

    predictions = loaded_model.predict(X.values)
    predictions_proba = loaded_model.predict_proba(X.values)

    result = pd.DataFrame(predictions_proba, index=df.index)
    result.to_csv('result_example.csv', index_label='id')

    # Метрики
    recall = recall_score(y_true=y, y_pred=predictions, average='weighted')
    f1 = f1_score(y_true=y, y_pred=predictions, average='weighted')
    precision = precision_score(y_true=y, y_pred=predictions, average='weighted')

    return recall, f1, precision


def feature_construct(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)

    # Преобразование месяцев
    ord_e = OrdinalEncoder(
        categories=[["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]])
    df["month"] = ord_e.fit_transform(df["month"].values.reshape(-1, 1))

    # Преобразование дней
    ord_e = OrdinalEncoder(categories=[["mon", "tue", "wed", "thu", "fri", "sat", "sun"]])
    df["day_of_week"] = ord_e.fit_transform(df["day_of_week"].values.reshape(-1, 1))

    # Принимаю решение удалить этот столбец
    df.drop(columns=["poutcome"], inplace=True)

    # Объединим single и divorced в просто single
    dict_to_replace = {"married": 1, "single": 0, "divorced": 0}
    df = df.replace({"marital": dict_to_replace})

    le = LabelEncoder()
    df['housing'] = le.fit_transform(df['housing'])
    df['loan'] = le.fit_transform(df['loan'])
    df['default'] = le.fit_transform(df['default'])

    # Бинаризация профессии: 1 - работает, 0 - не работает
    dict_to_replace = {"housemaid": 1, "services": 1, "admin.": 1, "blue-collar": 1, "technician": 1, "retired": 0,
                       "management": 1, "unemployed": 0, "self-employed": 1, "entrepreneur": 1, "student": 0}
    df = df.replace({"job": dict_to_replace})

    # Перевод образования в числовой формат: 1 - есть высшее, 0 - нет
    dict_to_replace = {"university.degree": 1, "high.school": 0, "basic.9y": 0, "professional.course": 1, "basic.4y": 0,
                       "basic.6y": 0}
    df = df.replace({"education": dict_to_replace})

    # Узнаем, сколько лежит людей в промежутках: (17-30), (30-50), (50-100) (young, adult, old)
    cut_labels = ['young', 'adult', 'old']
    cut_bins = [16, 30, 50, 200]

    df['age'] = pd.cut(df['age'],
                       bins=cut_bins,
                       labels=cut_labels)
    df['age'] = le.fit_transform(df['age'])

    # Обработаем pdays: 999 - значит, что не сталкивались с клиентом до этого времени
    cut_labels = ['old_client', 'new_client']
    cut_bins = [-1, 998, 1000]

    df['pdays'] = pd.cut(df['pdays'],
                         bins=cut_bins,
                         labels=cut_labels)
    df['pdays'] = le.fit_transform(df['pdays'])

    return df
