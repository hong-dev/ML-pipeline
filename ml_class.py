import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline


class TestSplit:
    # split dataset into train and validation subsets
    def split_test(self, dataset, target):
        self.features_without_target = dataset.drop(target, axis=1)
        self.target_feature = dataset[target]

        X_train, X_validation, y_train, y_validation = train_test_split(
            self.features_without_target, self.target_feature
        )

        return X_train, X_validation, y_train, y_validation


class Preprocessor(TransformerMixin):
    def __init__(self, scaler):
        self.scaler = scaler

    # transform numeric and categorical features
    def transform_columns(self):
        return ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    self.transform_numeric_features(),
                    self.numeric_features,
                ),
                (
                    "categorical",
                    self.transform_categorical_features(),
                    self.categorical_features,
                ),
            ]
        )

    # imputer and scaler for numeric features
    def transform_numeric_features(self):
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", self.scaler()),
            ]
        )

    # imputer and onehotencoder for categorical features
    @staticmethod
    def transform_categorical_features():
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder()),
            ]
        )

    # get column names for dataframe
    def get_column_names(self):
        cats_names = self.column_transformer.named_transformers_[
            "categorical"
        ]["onehot"].get_feature_names(self.categorical_features)
        feat_names = np.concatenate([self.numeric_features, cats_names])

        return feat_names

    def fit(self, X, y=None):
        self.numeric_features = X.select_dtypes(np.number).columns
        self.categorical_features = X.select_dtypes(exclude=np.number).columns

        self.column_transformer = self.transform_columns()
        self.column_transformer.fit(X)

        return self

    def transform(self, X):
        transformed_data = self.column_transformer.transform(X)
        column_names = self.get_column_names()

        return pd.DataFrame(
            transformed_data, columns=column_names, index=X.index
        )
