import os
import argparse
import pandas as pd
import numpy as np

from itertools import product
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)

import config
from utils import Preprocessor, FeatureSelector


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

args = get_arguments()

target = args.target

dataset_dir = "./data"
train_file_path = os.path.join(dataset_dir, args.train)
test_file_path = os.path.join(dataset_dir, args.input)

result_dir = "./result"
prediction_file_path = os.path.join(result_dir, args.prediction)
report_file_path = os.path.join(result_dir, args.report)

joblib_dir = "./joblib"
preprocessor_joblib_path = os.path.join(joblib_dir, "{}" + ".joblib")
model_joblib_path = os.path.join(joblib_dir, "{}, {}" + ".joblib")



def main():
    # create prediction dataframe and score dataframe list
    prediction_df = pd.DataFrame()
    score_df_list = []

    # transform, predict, score for each combination
    for scaler, (model_class, params) in product(
        config.scalers, config.models
    ):

        # assign preprocessor and model
        preprocessor = Preprocessor(scaler)
        model = model_class(**params)

        # process train, validation, test data
        train_score, validation_score = process_train_data(preprocessor, model)
        test_prediction, test_score = process_test_data(preprocessor, model)

        # add predicted data to prediction dataframe
        prediction_df[f"{scaler.__name__}-{model}"] = test_prediction

        # add score dataframes to the list
        score_df_list.extend([train_score, validation_score, test_score])

    # concatenate score dataframes
    report_df = pd.concat(score_df_list)

    # save prediction and report to csv files
    prediction_df.to_csv(prediction_file_path)
    report_df.to_csv(report_file_path)




def process_train_data(preprocessor, model):
    """
    return: score(train data, validation data)
    """

    # get dataset
    train_data = pd.read_csv(train_file_path)

    # split data
    (X_train, X_validation, y_train, y_validation) = split_test(
        train_data, target
    )

    pipe = Pipeline(
        [
            ("selector", FeatureSelector()),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipe.fit(X_train, y_train)

    pipe_path = model_joblib_path.format(
        model, preprocessor.scaler.__name__
    )
    dump(pipe, pipe_path)

    X_train_prediction = load(pipe_path).predict(X_train)
    X_validation_prediction = load(pipe_path).predict(X_validation)

    # score
    train_score = get_scores(
        y_train, X_train_prediction, preprocessor.scaler, model, "train",
    )
    validation_score = get_scores(
        y_validation,
        X_validation_prediction,
        preprocessor.scaler,
        model,
        "validation",
    )

    return train_score, validation_score


def process_test_data(preprocessor, model):
    """
    return: prediction, score
    """

    # get dataset
    test_data = pd.read_csv(test_file_path)

    pipe_path = model_joblib_path.format(
        model, preprocessor.scaler.__name__
    )
    test_prediction = load(pipe_path).predict(
        test_data.drop(target, axis=1)
    )

    # score
    drop_index = test_data[test_data[target].isnull()].index.tolist()
    test_score = get_scores(
        test_data[target].dropna(),
        np.delete(test_prediction, drop_index),
        preprocessor.scaler,
        model,
        "test",
    )

    return test_prediction, test_score


def split_test(dataset, target):
    X = dataset.drop(target, axis=1)
    y = dataset[target]

    X_train, X_validation, y_train, y_validation = train_test_split(X, y)

    return X_train, X_validation, y_train, y_validation


def get_scores(actual_y, predicted_y, scaler, model, data_name):
    """
    return: score dataframe
    """

    score_functions = [
        precision_score,
        recall_score,
        accuracy_score,
        f1_score,
    ]
    scores = []

    for function in score_functions:
        scores.append(function(actual_y, predicted_y))

    # get multi-index for report dataframe
    report_index = pd.MultiIndex.from_tuples(
        [(scaler.__name__, model, data_name)],
        names=["scaler", "model", "data"],
    )

    # create dataframe for scores
    score_data = pd.DataFrame(
        [scores],
        columns=[function.__name__ for function in score_functions],
        index=report_index,
    )

    return score_data


if __name__ == "__main__":
    main()
