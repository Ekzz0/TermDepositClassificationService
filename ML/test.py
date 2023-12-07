import pandas as pd
import utils as ut
from utils.json_scripts import convert_dataframe_to_json, convert_json_to_dataframe


if __name__ == '__main__':
    path = "./data/test.csv"
    df = pd.read_csv(path, index_col=0)
    X, y = ut.split_to_x_y(df, 'y')

    # Генерация признаков
    # Отсюда имитируем ситуацию: Получили json файл
    X = convert_dataframe_to_json(X)

    # 1) Нужно конвертировать json в pd.DataFrame
    X = convert_json_to_dataframe(X)
    # 2) Импортируем конструктор признаков
    feature_construct = ut.load_feature_constructor()
    # 3) Получаем обработанный pd.DataFrame
    X_test = feature_construct(X)
    # 4) Загружаем модель
    path = "./models/RandomForest.pkl"
    model = ut.load_model(path)
    # 5) Получаем предикт
    clf_report = model.predict(X_test)
    # 6) Конвертируем в json и отдаем обратно
    response = convert_dataframe_to_json(clf_report)
    print(response)

    # # Получим топ фич, которые повлияли на ответ
    # ID = 37202
    # print(model.get_feature_importance(X_test, ID))
    #
    # # Обучение модели на каких-то новых данных:
    # path = "./data/train.csv"
    # train_data = pd.read_csv(path, index_col=0)
    # X, y = ut.split_to_x_y(train_data, 'y')
    # # Отсюда имитируем ситуацию: Получили json файл
    # X = convert_dataframe_to_json(X)
    # y = convert_dataframe_to_json(y)
    #
    # # 1) Нужно конвертировать json в pd.DataFrame
    # X = convert_json_to_dataframe(X)
    # y = convert_json_to_dataframe(y)
    # # 2) Обучаем модель
    # score = model.fit(X, y)
    # print(score)


