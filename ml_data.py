import os
import argparse
import pandas as pd

from itertools import product
from collections import namedtuple
from joblib import dump, load

from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    MaxAbsScaler,
)
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from ml_class import TestSplit, Preprocessor

joblib_dir = "./joblib/"
preprocessor_joblib = os.path.join(joblib_dir, "{}" + ".joblib")
model_joblib = os.path.join(joblib_dir, "{}" + ".joblib")


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        default="marketing_train.csv",
        help="Write file name you want to train",
    )
    parser.add_argument(
        "--input",
        default="marketing_test.csv",
        help="Write file name you want to predict",
    )
    parser.add_argument(
        "--prediction",
        default="pred.csv",
        help="Write file name you want to save prediction as",
    )
    parser.add_argument(
        "--report",
        default="report.csv",
        help="Write file name you want to save report as",
    )
    parser.add_argument(
        "--target",
        default="insurance_subscribe",
        help="Write target feature",
    )

    return parser.parse_args()


def main():
    scalers = [
        StandardScaler,
        RobustScaler,
        MinMaxScaler,
        MaxAbsScaler,
    ]

    models = {
        LogisticRegression: {"max_iter": 700},
        DecisionTreeClassifier: {"max_depth": 4},
        RandomForestClassifier: {"max_depth": 4},
        LinearSVC: {"dual": False},
    }

    # get combinations of scalers and models
    scaler_model_combination = list(product(scalers, models))
    named_tuple = namedtuple("Index", "scaler model")
    index_combination = [
        named_tuple(index[0], index[1]) for index in scaler_model_combination
    ]

    # create prediction dataframe and score dataframe list
    prediction_df = pd.DataFrame()
    score_df_list = []

    # transform, predict, score for each combination
    for index in index_combination:

        # assign preprocessor and model
        preprocessor = Preprocessor(index.scaler)
        model = index.model(**models[index.model])

        # process train, validation, test data
        train_score, validation_score = process_train_data(
            preprocessor, model
        )
        test_prediction, test_score = process_test_data(preprocessor, model)

        # add predicted data to prediction dataframe
        prediction_df[
            f"{index.scaler.__name__}-{index.model.__name__}"
        ] = test_prediction

        # add score dataframes to the list
        # scores = {
        #     "train": train_score,
        #     "validation": validation_score,
        #     "test": test_score,
        # }
        # score_df = get_score_dataframe(index.scaler, index.model, scores)
        # score_df_list.append(score_df)
        score_df_list.extend([train_score, validation_score, test_score])

    # concatenate score dataframes
    report_df = pd.concat(score_df_list)

    # save prediction and report to csv files
    arguments = get_arguments()
    result_dir = "./result"

    prediction_df.to_csv(os.path.join(result_dir, arguments.prediction))
    report_df.to_csv(os.path.join(result_dir, arguments.report))


def process_train_data(preprocessor, model):
    """
    return: score(train data, validation data)
    """

    # get dataset
    arguments = get_arguments()
    train_data = pd.read_csv(os.path.join("./data", arguments.train))

    # split data
    target = get_arguments().target
    (X_train, X_validation, y_train, y_validation) = TestSplit().split_test(
        train_data, target
    )

    # fit preprocessor
    preprocessor.fit(X_train)
    preprocessor_path = preprocessor_joblib.format(
        preprocessor.scaler.__name__
    )
    dump(preprocessor, preprocessor_path)

    # transform preprocessor
    X_train_transformed = load(preprocessor_path).transform(X_train)
    X_validation_transformed = load(preprocessor_path).transform(X_validation)

    # fit model
    model.fit(X_train_transformed, y_train)
    model_path = model_joblib.format(model)
    dump(model, model_path)

    # predict
    X_train_prediction = load(model_path).predict(X_train_transformed)
    X_validation_prediction = load(model_path).predict(
        X_validation_transformed
    )

    # score
    train_score = get_scores(y_train, X_train_prediction)
    validation_score = get_scores(y_validation, X_validation_prediction)

    #######
    a = get_score_dataframe(preprocessor.scaler, model, "train", train_score)
    b = get_score_dataframe(preprocessor.scaler, model, "validation", validation_score)

    # return train_score, validation_score
    return a, b


def process_test_data(preprocessor, model):
    """
    return: prediction, score
    """

    # get dataset
    arguments = get_arguments()
    test_data = pd.read_csv(os.path.join("./data", arguments.input))
    target = arguments.target

    # transform
    preprocessor_path = preprocessor_joblib.format(
        preprocessor.scaler.__name__
    )
    test_transformed = load(preprocessor_path).transform(
        test_data.drop(target, axis=1)
    )

    # predict
    model_path = model_joblib.format(model)
    test_prediction = load(model_path).predict(test_transformed)

    # score
    test_score = get_scores(test_data[target], test_prediction)

    ##############
    a = get_score_dataframe(preprocessor.scaler, model, "test", test_score)

    # return test_prediction, test_score
    return test_prediction, a


def get_scores(actual_y, predicted_y):
    score_functions = [
        precision_score,
        recall_score,
        accuracy_score,
        f1_score,
    ]
    scores = []

    for function in score_functions:
        scores.append(function(actual_y, predicted_y))

    return scores


def get_score_dataframe(scaler, model, data_name, scores):

    # get multi-index for report dataframe
    report_index = pd.MultiIndex.from_tuples(
        [(scaler.__name__, model, data_name)],
        names=["scaler", "model", "data"],
    )

    # create dataframe for scores
    score_data = pd.DataFrame(
        [scores],
        columns=["Precision", "Recall", "Accuracy", "F1"],
        index=report_index,
    )

    return score_data
 
# def get_score_dataframe(scaler, model, scores):
#     for data_name, score in scores.items():

#         # get multi-index for report dataframe
#         report_index = pd.MultiIndex.from_tuples(
#             [(scaler.__name__, model.__name__, data_name)],
#             names=["scaler", "model", "data"],
#         )

#         # create dataframe for scores
#         score_data = pd.DataFrame(
#             [score],
#             columns=["Precision", "Recall", "Accuracy", "F1"],
#             index=report_index,
#         )

#     return score_data
# 


if __name__ == "__main__":
    main()
