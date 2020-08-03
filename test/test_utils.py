import unittest
import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    MaxAbsScaler,
)

from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from utils import Preprocessor, FeatureSelector

class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.scalers = [
            StandardScaler,
            MinMaxScaler,
            RobustScaler,
            MaxAbsScaler,
        ]

    def _run_preprocessor(self, data, scaler):
        dataset = pd.DataFrame(data=data)

        feature_selector = FeatureSelector()
        dataset = feature_selector.fit_transform(dataset)

        preprocessor = Preprocessor(scaler=scaler)
        fitted_preprocessor = preprocessor.fit_transform(dataset)

        row_length = len(dataset)
        numerical_features = dataset.select_dtypes(np.number)
        categorical_features = dataset.select_dtypes(exclude=np.number)

        imputer = SimpleImputer(strategy="most_frequent")
        imputed_values = imputer.fit_transform(categorical_features)
        imputed_df = pd.DataFrame(imputed_values)

        column_length = len(numerical_features.columns)
        for key in imputed_df:
            column_length += len(set(imputed_df[key]))

        self.assertEqual(
            fitted_preprocessor.shape, (row_length, column_length)
        )
        self.assertEqual(fitted_preprocessor.isnull().values.any(), False)
        self.assertEqual(
            fitted_preprocessor.drop(numerical_features, axis=1)
            .isin([0, 1])
            .all()
            .all(),
            True,
        )

    def test_small_data(self):
        """
        dataset: two numeric features, two categorical features
        """

        data = {
            "num1": [1, 2, 3],
            "cat1": ["a", "b", "c"],
            "num2": [4, 5, 6],
            "cat2": ["d", "e", "f"],
        }

        for scaler in self.scalers:
            self._run_preprocessor(data, scaler)

    def test_outlier(self):
        """
        dataset: outlier in numeric feature
        """

        data = {
            "num1": [1, 2, 9999999999],
            "cat1": ["a", "b", "c"],
            "num2": [4, 5, 6],
            "cat2": ["d", "e", "f"],
        }

        for scaler in self.scalers:
            self._run_preprocessor(data, scaler)

    def test_negative_num(self):
        """
        dataset: negative number in numeric feature
        """

        data = {
            "num1": [1, 2, 3],
            "cat1": ["a", "b", "c"],
            "num2": [4, -50, 6],
            "cat2": ["d", "e", "f"],
        }

        for scaler in self.scalers:
            self._run_preprocessor(data, scaler)

    def test_missing_numeric(self):
        """
        dataset: missing values in numeric feature
        """

        data = {
            "num1": [1, 2, None],
            "cat1": ["a", "b", "c"],
            "num2": [4, None, 6],
            "cat2": ["d", "e", "f"],
        }

        for scaler in self.scalers:
            self._run_preprocessor(data, scaler)

    def test_missing_categorical(self):
        """
        dataset: missing values in categorical feature
        """

        data = {
            "num1": [1, 2, 3],
            "cat1": ["a", "b", np.NaN],
            "num2": [4, 5, 6],
            "cat2": ["d", np.NaN, "f"],
        }

        for scaler in self.scalers:
            self._run_preprocessor(data, scaler)

    def test_missing_num_cat_values(self):
        """
        dataset: missing values in numeric and categorical features
        """

        data = {
            "num1": [None, 2, 3],
            "cat1": ["a", np.NaN, np.NaN],
            "num2": [None, None, 6],
            "cat2": ["d", np.NaN, "f"],
        }

        for scaler in self.scalers:
            self._run_preprocessor(data, scaler)

    def test_titanic_data(self):
        """
        dataset: actual csv data (titanic)
        """

        data = pd.read_csv("./data/source_kaggle_titanic_train.csv")

        for scaler in self.scalers:
            self._run_preprocessor(data, scaler)

    def test_marketing_data(self):
        """
        dataset: actual csv data (marketing)
        """

        data = pd.read_csv("./data/marketing_train.csv")

        for scaler in self.scalers:
            self._run_preprocessor(data, scaler)


if __name__ == "__main__":
    unittest.main(exit=False)
