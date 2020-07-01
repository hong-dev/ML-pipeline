import argparse
import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LogisticRegression, LinearRegression

# from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

class TestSplit:
    def split_test(self, dataset, target):
        self.features_without_target = dataset.drop(target, axis=1)
        self.target_feature = dataset[target]

        X_train, X_test, y_train, y_test = train_test_split(self.features_without_target, self.target_feature) #기본비율
        return X_train, X_test, y_train, y_test
        
class Preprocessor(TransformerMixin):
    def __init__(self, scaler):
        self.scaler = scaler

    def transform_numeric_features(self):
        return Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', self.scaler())
            ])

    @staticmethod
    def transform_categorical_features():
        return Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder())
            ])

    def transform_columns(self):
        return ColumnTransformer(transformers = [
            ('nums', self.transform_numeric_features(), self.numeric_features.columns),
            ('cats', self.transform_categorical_features(), self.categorical_features.columns)
            ])

    def get_column_names(self):
        cats_names = self.column_transformer.named_transformers_['cats']['onehot'].get_feature_names(self.categorical_features.columns)
        feat_names = np.concatenate([self.numeric_features.columns, cats_names])
        return feat_names

    def fit(self, X, y=None):
        self.dataset = X
        self.numeric_features = X.select_dtypes(np.number)
        self.categorical_features = X.select_dtypes(exclude=np.number)
        self.column_transformer = self.transform_columns()
        self.column_transformer.fit(X)
        return self
    
    def transform(self, X):
        transformed_data = self.column_transformer.transform(X)
        column_names = self.get_column_names()
        return pd.DataFrame(transformed_data, columns=column_names, index=X.index)