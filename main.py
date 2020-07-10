import pandas as pd

from itertools import product
from joblib import dump, load
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)

from utils import Preprocessor
from config import (
    scalers,
    models,
    target,
    train_file,
    test_file,
    preprocessor_joblib,
    model_joblib,
    prediction_file,
    report_file,
)


def main():
    # create prediction dataframe and score dataframe list
    prediction_df = pd.DataFrame()
    score_df_list = []

    # transform, predict, score for each combination
    for scaler, (model_class, params) in product(scalers, models):

        # assign preprocessor and model
        preprocessor = Preprocessor(scaler)
        model = model_class(**params)

        # process train, validation, test data
        train_score, validation_score = process_train_data(preprocessor, model)
        test_prediction, test_score = process_test_data(preprocessor, model)

        # add predicted data to prediction dataframe
        prediction_df[
            f"{scaler.__name__}-{model}"
        ] = test_prediction

        # add score dataframes to the list
        score_df_list.extend([train_score, validation_score, test_score])

    # concatenate score dataframes
    report_df = pd.concat(score_df_list)

    # save prediction and report to csv files
    prediction_df.to_csv(prediction_file)
    report_df.to_csv(report_file)


def process_train_data(preprocessor, model):
    """
    return: score(train data, validation data)
    """

    # get dataset
    train_data = pd.read_csv(train_file)

    # split data
    (X_train, X_validation, y_train, y_validation) = split_test(
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
    test_data = pd.read_csv(test_file)

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
    test_score = get_scores(
        test_data[target], test_prediction, preprocessor.scaler, model, "test"
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
