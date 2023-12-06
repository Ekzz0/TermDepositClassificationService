import joblib
from sklearn.metrics import recall_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from .data_processing import get_feature_indexes, get_importance
from .data_structures import ModelPrediction, Person, PersonsList, FeatureImportance, Score
import pandas as pd
import numpy as np


class LoadedModel:
    bin_f = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'pdays']
    count_f = ['age', 'month', 'day_of_week']
    num_f = ['duration', 'campaign', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m',
             'nr.employed']

    def __init__(self, path: str):
        # Загрузка модель из файла
        self.model = joblib.load(path)
        self.path = path

    def predict(self, X: pd.DataFrame) -> ModelPrediction:
        pred = pd.DataFrame(self.model.predict_proba(X.values),
                            index=X.index,
                            columns=['Не оформил', 'Оформил'])

        pred.index.rename('id', inplace=True)

        persons = [Person(probability=probability, ID=ID) for ID, probability in
                   zip(pred.index, pred['Оформил'].values)]

        result = PersonsList(persons=persons)

        return ModelPrediction(result=result)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Score:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train = X_train if type(X_train) == np.ndarray else X_train.values
        y_train = y_train if type(y_train) == np.ndarray else y_train.values.ravel()
        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test.values)

        # Метрики
        recall = recall_score(y_true=y_test.values.ravel(), y_pred=predictions, average='weighted')
        f1 = f1_score(y_true=y_test.values.ravel(), y_pred=predictions, average='weighted')
        precision = precision_score(y_true=y_test.values.ravel(), y_pred=predictions, average='weighted')

        return Score(recall=recall, f1=f1, precision=precision)

    def save_model(self):
        joblib.dump(self.model, self.path)

    def get_feature_importance(self, df: pd.DataFrame, ID: int) -> FeatureImportance:
        try:
            fi = self.model.feature_importances_
        except Exception as e:
            print('Модель еще не обучена.')
        else:
            # Получим индексы для всех типов признаков
            bin_f_ind = get_feature_indexes(df.columns, self.bin_f)
            count_f_ind = get_feature_indexes(df.columns, self.count_f)
            num_f_ind = get_feature_indexes(df.columns, self.num_f)
            try:
                # Получим влияние признаков для конкретного пользователя
                bin_f_imp = get_importance(df, fi, ID, bin_f_ind)
                count_f_imp = get_importance(df, fi, ID, count_f_ind)
                num_f_imp = get_importance(df, fi, ID, num_f_ind)
            except Exception as e:
                print('Датасет не обработан полностью')
            else:
                # Индекс конца датасета
                last_ind_b = len(bin_f_imp)
                last_ind_c = len(count_f_imp)
                last_ind_n = len(num_f_imp)

                # Топ 2 признака для каждого типа
                top_bin = bin_f_imp.iloc[last_ind_b - 2:last_ind_b]
                top_count = count_f_imp.iloc[last_ind_c - 2:last_ind_c]
                to_num = num_f_imp.iloc[last_ind_n - 2:last_ind_n]

                # Формируем отчет
                feature_names = list(top_bin.index) + list(top_count.index) + list(to_num.index)
                feature_importance = list(top_bin.values.ravel()) + list(top_count.values.ravel()) + list(
                    to_num.values.ravel())

                return FeatureImportance(features=feature_names, importance=feature_importance)
