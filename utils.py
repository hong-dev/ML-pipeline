import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
from sklearn.pipeline import make_pipeline


class Preprocessor(TransformerMixin):
    def __init__(self, scaler):
        self.scaler = scaler
        self.numeric_features = None
        self.categorical_features = None

    # preprocess numeric and categorical features
    def _build_pipeline(self):
        return ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    make_pipeline(
                        SimpleImputer(strategy="median"), self.scaler()
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

    # get column names for dataframe
    def get_column_names(self):
        cats_names = self.column_transformer.named_transformers_[
            "categorical"
        ][1].get_feature_names(self.categorical_features)
        feat_names = np.concatenate([self.numeric_features, cats_names])

        return feat_names

    def fit(self, X, y=None):
        self.numeric_features = X.select_dtypes(np.number).columns
        self.categorical_features = X.select_dtypes(exclude=np.number).columns

        self.column_transformer = self._build_pipeline()
        self.column_transformer.fit(X)

        return self

    def transform(self, X):
        transformed_data = self.column_transformer.transform(X)
        column_names = self.get_column_names()

        return pd.DataFrame(
            transformed_data, columns=column_names, index=X.index
        )
