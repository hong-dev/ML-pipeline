import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
from sklearn.pipeline import make_pipeline

class FeatureSelector(TransformerMixin):
    def __init__(self):
        self.remove_columns = None

    @staticmethod
    def _remove_features(X):
        remove_columns = []
        for column in X:
            if (X[column].dtype == object) and (len(X[column].unique()) > 30):
                remove_columns.append(column)

        return remove_columns

    def fit(self, X, y=None):
        self.remove_columns = self._remove_features(X)

        return self

    def transform(self, X):
        return X.drop(columns=self.remove_columns)


class Preprocessor(TransformerMixin):
    def __init__(self, scaler):
        self.scaler = scaler
        self.numeric_features = None
        self.categorical_features = None
        self.pipeline = None

    def _set_feature_types(self, X):
        self.numeric_features = X.select_dtypes(np.number).columns
        self.categorical_features = X.select_dtypes(exclude=np.number).columns
        return

    # preprocess numeric and categorical features
    def _build_pipeline(self):
        assert (
            self.numeric_features is not None
            and self.categorical_features is not None
        )

        self.pipeline = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    make_pipeline(
                        SimpleImputer(strategy="median"), self.scaler(),
                    ),
                    self.numeric_features,
                ),
                (
                    "categorical",
                    make_pipeline(
                        SimpleImputer(strategy="most_frequent"),
                        OneHotEncoder(),
                    ),
                    self.categorical_features,
                ),
            ]
        )
        return

    # get column names for dataframe
    def get_column_names(self):
        cats_names = self.pipeline.named_transformers_["categorical"][
            1
        ].get_feature_names(self.categorical_features)
        feat_names = np.concatenate([self.numeric_features, cats_names])

        return feat_names

    def fit(self, X, y=None):
        self._set_feature_types(X)
        self._build_pipeline()
        self.pipeline.fit(X)

        return self

    def transform(self, X):
        transformed_data = self.pipeline.transform(X)
        column_names = self.get_column_names()

        return pd.DataFrame(
            transformed_data, columns=column_names, index=X.index,
        )
