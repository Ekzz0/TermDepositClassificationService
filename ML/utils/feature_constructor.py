from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import pandas as pd


class FeatureConstructor:
    def __init__(self):
        self.df: pd.DataFrame = pd.DataFrame()

    def feature_construct(self, path) -> pd.DataFrame:
        self.df = pd.read_csv(path, index_col=0)
        # Преобразование месяцев
        ord_e = OrdinalEncoder(
            categories=[["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]])
        self.df["month"] = ord_e.fit_transform(self.df["month"].values.reshape(-1, 1))

        # Преобразование дней
        ord_e = OrdinalEncoder(categories=[["mon", "tue", "wed", "thu", "fri", "sat", "sun"]])
        self.df["day_of_week"] = ord_e.fit_transform(self.df["day_of_week"].values.reshape(-1, 1))

        # Принимаю решение удалить этот столбец
        self.df.drop(columns=["poutcome"], inplace=True)

        # Объединим single и divorced в просто single
        dict_to_replace = {"married": 1, "single": 0, "divorced": 0}
        self.df = self.df.replace({"marital": dict_to_replace})

        le = LabelEncoder()
        self.df['housing'] = le.fit_transform(self.df['housing'])
        self.df['loan'] = le.fit_transform(self.df['loan'])
        self.df['default'] = le.fit_transform(self.df['default'])

        # Бинаризация профессии: 1 - работает, 0 - не работает
        dict_to_replace = {"housemaid": 1, "services": 1, "admin.": 1, "blue-collar": 1, "technician": 1, "retired": 0,
                           "management": 1, "unemployed": 0, "self-employed": 1, "entrepreneur": 1, "student": 0}
        self.df = self.df.replace({"job": dict_to_replace})

        # Перевод образования в числовой формат: 1 - есть высшее, 0 - нет
        dict_to_replace = {"university.degree": 1, "high.school": 0, "basic.9y": 0, "professional.course": 1,
                           "basic.4y": 0,
                           "basic.6y": 0}
        self.df = self.df.replace({"education": dict_to_replace})

        # Узнаем, сколько лежит людей в промежутках: (17-30), (30-50), (50-100) (young, adult, old)
        cut_labels = ['young', 'adult', 'old']
        cut_bins = [16, 30, 50, 200]

        self.df['age'] = pd.cut(self.df['age'],
                           bins=cut_bins,
                           labels=cut_labels)
        self.df['age'] = le.fit_transform(self.df['age'])

        # Обработаем pdays: 999 - значит, что не сталкивались с клиентом до этого времени
        cut_labels = ['old_client', 'new_client']
        cut_bins = [-1, 998, 1000]

        self.df['pdays'] = pd.cut(self.df['pdays'],
                             bins=cut_bins,
                             labels=cut_labels)
        self.df['pdays'] = le.fit_transform(self.df['pdays'])
        self.df.to_csv('./data/test_ready.csv')
        return self.df


