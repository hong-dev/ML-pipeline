import argparse
import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler as SklearnRobustScaler
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from sklearn.impute import SimpleImputer as SklearnSimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

class StandardScaler(SklearnStandardScaler):
    def transform(self, X):
        return pd.DataFrame(super().transform(X), columns=X.columns, index=X.index)

class RobustScaler(SklearnRobustScaler):
    def transform(self, X):
        return pd.DataFrame(super().transform(X), columns=X.columns, index=X.index)

class SimpleImputer(SklearnSimpleImputer):
    def transform(self, X):
        return pd.DataFrame(super().transform(X), columns=X.columns, index=X.index)

class OneHotEncoder(SklearnOneHotEncoder):
    def transform(self, X):
        sparse_matrix = super(OneHotEncoder, self).transform(X)
        new_columns = self.get_column_names(X=X)
        return pd.DataFrame(sparse_matrix.toarray(), columns=new_columns, index=X.index)

    def get_column_names(self, X):
        new_columns = []
        for i, column in enumerate(X.columns):
            j = 0
            while j < len(self.categories_[i]):
                new_columns.append(f'{column}_{self.categories_[i][j]}')
                j += 1
        return new_columns
        
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
        nums_names = self.column_transformer.named_transformers_['nums'].fit_transform(self.numeric_features).columns
        cats_names = self.column_transformer.named_transformers_['cats'].fit_transform(self.categorical_features).columns
        feat_names = np.concatenate([nums_names, cats_names])
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


# train_data = pd.read_csv('./data/marketing_train.csv')
# test_data = pd.read_csv('./data/marketing_test.csv')

# target = "insurance_subscribe"
# X_train, X_val, y_train, y_val = TestSplit().split_test(train_data, target)

# preprocessor = Preprocessor()
# X_train_transformed = preprocessor.fit_transform(X_train)
# X_test_transformed = preprocessor.transform(X_val)
# test_transformed = preprocessor.transform(test_data.drop(target, axis=1))


# model = LogisticRegression(max_iter=600)
# model.fit(X_train_transformed, y_train)

# X_train_prediction = model.predict(X_train_transformed)
# X_validation_prediction = model.predict(X_test_transformed)
# test_prediction = model.predict(test_transformed)

# X_train_prediction = predict(X_train_transformed)
# X_validation_prediction = predict(X_test_transformed)
# test_prediction = predict(test_transformed)

# def predict(transformed_data):
#     return model.predict(transformed_data)

# def make_pred():
#     prediction_df = pd.DataFrame(test_prediction, columns=["Predicted value"]).rename_axis("ID")
#     prediction_df.to_csv("pred.csv")


def get_scores(actual_y, predicted_y):
    funcs = [precision_score, recall_score, accuracy_score, f1_score]
    # columns = []
    scores = []

    for func in funcs:
        # columns.append(func.__name__)
        scores.append(func(actual_y, predicted_y))

    # report_df = pd.DataFrame(columns=columns)
    # report_df.loc['Robust-Logistic'] = scores

    return scores


def make_report():
    report_df = pd.DataFrame(index=["Logistic"], columns=["Precision", "Recall", "Accuracy", "F1"])

    compare_y = [(y_train, X_train_prediction), (y_val, X_validation_prediction), (test_data[target], test_prediction)]
    for y_data in compare_y:
        actual_y, predicted_y = y_data[0], y_data[1]
        # report_df.loc[('Logistic', 'train')] = get_scores(actual_y, predicted_y)
        
    print(report_df)
    return report_df

# make_report()

# print(get_scores(y_train, X_train_prediction))
# print(get_scores(y_test, X_test_prediction))
# print(get_scores(test_data[target], test_prediction))

# get_scores.to_csv("report.csv")


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input', required=True)
#     parser.add_argument('--prediction', default='pred.csv', help='Write file name you want to save prediction')
#     parser.add_argument('--report', default='report.csv', help='Write file name you want to save report')

#     train, test = read_data()
#     transformed_data = preprocess()
#     model  = model_train()

# def read_data(path='marketing_train.csv'):
#     return pd.read_csv(path)

# def test():
#     read_data(input_path)

# 1. train 하기
# 2. input으로 들어온 데이터 읽기
# 3. 데이터 transform, prediction, score  하기
# 4. 저장하기 

# if __name__ == "__main__":
#     main()


# k-fold, model seletor