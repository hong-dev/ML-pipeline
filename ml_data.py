import argparse
import pandas as pd

from collections import namedtuple
from itertools import product

from ml_class import TestSplit, Preprocessor#, RobustScaler, StandardScaler

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def main():
    scalers = [RobustScaler, StandardScaler, MinMaxScaler, MaxAbsScaler]
    # models = [LogisticRegression, DecisionTreeClassifier]

    models = {
        LogisticRegression: {"max_iter": 700},
        DecisionTreeClassifier: {"max_depth": 4}
    }

    # dataset_type = ['train', 'validation', 'test']

    # comb = list(product(scalers, models))

    # scalers_models_combination = list(product(scalers, models))
    index_combination = list(product(scalers, models))

    a = namedtuple('Index', 'scaler model')
    index_combination_a = [a(index[0], index[1]) for index in index_combination]

    # print(scalers_models_combination)
    
    # comb_n = []
    # for index in scalers_models_combination:
        # scaler = index[0]
        # model = index[1]

        # combination_name = f"{scaler.__name__}-{model.__name__}"
        # comb_n.append(combination_name)
    
    # a = list(product(comb_n, dataset_type))

    # index = pd.MultiIndex.from_tuples(index_combination_a, names=["scaler", "model"])

    # report_df = pd.DataFrame(index=index, columns=["Precision", "Recall", "Accuracy", "F1"])

    report_df = pd.DataFrame()
    prediction_df = pd.DataFrame()

    for index in index_combination_a:
        # scaler = index[0]
        # model = index[1]
        # combination_name = f"{scaler.__name__}-{model.__name__}"

        preprocessor = Preprocessor(index.scaler)
        model = index.model(**models[index.model])#(max_iter=700, max_depth=4)

        train_score, validation_score = process_train_data(preprocessor, model)
        test_prediction, test_score = process_test_data(preprocessor, model)
        prediction_df[f"{index.scaler.__name__}-{index.model.__name__}"] = test_prediction

        dataset_type = {'train': train_score, 'validation': validation_score, 'test': test_score}

        report_df = add_report(report_df, index.scaler, index.model, dataset_type)
        
        # report_df = report_df.append(add_report(report_df, index.scaler, index.model, dataset_type))

        print(f"finish {index}")

    # print(report_df)

    prediction_df.to_csv('./result/{}'.format(get_arguments().prediction))
    report_df.to_csv('./result/{}'.format(get_arguments().report))

def process_train_data(preprocessor, model):
    # get dataset
    train_data = get_data('marketing_train.csv')

    # split data
    target = get_arguments().target
    X_train, X_validation, y_train, y_validation = TestSplit().split_test(train_data, target)

    # fit & transform
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_validation_transformed = transform_data(preprocessor, X_validation)

    # fit & predict
    model.fit(X_train_transformed, y_train)
    X_train_prediction = predict_data(model, X_train_transformed)
    X_validatrion_prediction = predict_data(model, X_validation_transformed)

    # score
    train_score = get_scores(y_train, X_train_prediction)
    validation_score = get_scores(y_validation, X_validatrion_prediction)

    return train_score, validation_score

def process_test_data(preprocessor, model):
    """
    return: prediction, score
    """

    # get dataset
    arguments = get_arguments()
    test_data = get_data(arguments.input)
    target = arguments.target

    # transform
    test_transformed = transform_data(preprocessor, test_data.drop(target, axis=1))

    # predict
    test_prediction = predict_data(model, test_transformed)
        
    # score
    test_score = get_scores(test_data[target], test_prediction)

    return test_prediction, test_score

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='marketing_test.csv', help='Write file name you want to predict')
    parser.add_argument('--prediction', default='pred.csv', help='Write file name you want to save prediction')
    parser.add_argument('--report', default='report.csv', help='Write file name you want to save report')
    parser.add_argument('--target', default='insurance_subscribe', help='Write target feature')

    return parser.parse_args()

def get_data(file_name):
    return pd.read_csv('./data/{}'.format(file_name))

def transform_data(preprocessor, dataset_without_target):
    return preprocessor.transform(dataset_without_target)

def predict_data(model, transformed_data):
    return model.predict(transformed_data)

def get_scores(actual_y, predicted_y):
    funcs = [precision_score, recall_score, accuracy_score, f1_score]
    scores = []

    for func in funcs:
        scores.append(func(actual_y, predicted_y))

    return scores

def add_report(report, scaler, model, dataset_type):
    # dataset_type = {'train': train_score, 'validation': validation_score, 'test': test_score}

    # print(report)

    for data, score in dataset_type.items():
        index = pd.MultiIndex.from_tuples([(scaler.__name__, model.__name__, data)], names=["scaler", "model", "data"])
        # c = pd.Series(score, index=["Precision", "Recall", "Accuracy", "F1"], name=index) #(scaler.__name__, model.__name__, data))
        # c = report.append((score, index=report.columns))
        # print(c)
        d = pd.DataFrame([score], columns=["Precision", "Recall", "Accuracy", "F1"], index=index)
        # print(d)

        report = report.append(d)
        # print(report)
        # report.loc[scaler, model, data] = score

    return report

if __name__ == "__main__":
    main()

# python ml_data.py --input “marketing_test.csv” --prediction “pred.csv” --report “report.csv”
