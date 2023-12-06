import pandas as pd
import utils as ut

if __name__ == '__main__':
    # Генерация признаков
    path = "./data/test.csv"
    FeatureConstruct = ut.load_feature_constructor()
    test_data = FeatureConstruct.feature_construct(path)

    # Загрузка модели
    path = "./models/RandomForest.pkl"
    model = ut.load_model(path)

    # # Обучение модели на каких-то новых данных:
    # path = "./data/train.csv"
    # train_data = pd.read_csv(path, index_col=0)
    # X, y = ut.split_to_x_y(train_data, 'y')
    # score = model.fit(X, y)
    # print(score)
    # model.save_model()

    # Получение предикта
    X_pred, y_pred = ut.split_to_x_y(test_data, 'y')
    clf_report = model.predict(X_pred)
    result = clf_report.result
    # clf_report.predict.to_csv('../result.csv', index_label='id')

    # Получим топ фич, которые повлияли на ответ
    ID = 22720  # 323, 6605

    imp_test = test_data.drop(columns='y')
    print(model.get_feature_importance(imp_test, ID))
