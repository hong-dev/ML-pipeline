import os
import argparse

from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    MaxAbsScaler,
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


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


scalers = [
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    MaxAbsScaler,
]

models = [
    (LogisticRegression, {"max_iter": 700}),
    (DecisionTreeClassifier, {"max_depth": 4}),
    (RandomForestClassifier, {"max_depth": 4}),
    (LinearSVC, {"dual": False}),
]

args = get_arguments()

target = args.target

dataset_dir = "./data"
train_file = os.path.join(dataset_dir, args.train)
test_file = os.path.join(dataset_dir, args.input)

result_dir = "./result"
prediction_file = os.path.join(result_dir, args.prediction)
report_file = os.path.join(result_dir, args.report)

joblib_dir = "./joblib"
preprocessor_joblib = os.path.join(joblib_dir, "{}" + ".joblib")
model_joblib = os.path.join(joblib_dir, "{}" + ".joblib")
