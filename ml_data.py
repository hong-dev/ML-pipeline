import pandas as pd
import numpy as np

from sklearn.compose import make_column_transformer, ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import RobustScaler as SklearnRobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer as SklearnSimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin

from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from sklearn.compose import ColumnTransformer as SklearnColumnTransformer

class RobustScaler(SklearnRobustScaler):
    def transform(self, X):
        return pd.DataFrame(super().transform(X), columns=X.columns, index=X.index)

class SimpleImputer(SklearnSimpleImputer):
    def transform(self, X):
        return pd.DataFrame(super().transform(X), columns=X.columns, index=X.index)

class OneHotEncoder(SklearnOneHotEncoder):
    def transform(self, X):
        new_columns = self.get_new_columns(X=X)
        return pd.DataFrame(super().transform(X).toarray(), columns=new_columns, index=X.index)

    def get_new_columns(self, X):
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

        X_train, X_test, y_train, y_test = train_test_split(self.features_without_target, self.target_feature)
        return X_train, X_test, y_train, y_test

class Preprocessor(ColumnTransformer):
    def __init__(self, dataset):
        self.numeric_features = dataset.select_dtypes(np.number)
        self.categorical_features = dataset.select_dtypes(exclude=np.number)
        super().__init__(transformers= [
            ('nums', self.transform_numeric_features(), self.numeric_features.columns),
            ('cats', self.transform_categorical_features(), self.categorical_features.columns)
        ]

    @staticmethod
    def transform_numeric_features():
        return Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
            ])

    @staticmethod
    def transform_categorical_features():
        return Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder())
            ])

    # def transform_columns(self):
    #     return ColumnTransformer(transformers = [
    #         ('nums', self.transform_numeric_features(), self.numeric_features.columns),
    #         ('cats', self.transform_categorical_features(), self.categorical_features.columns)
    #         ])

    def get_column_names(self):
        nums_names = self.transformers['nums'].fit_transform(self.numeric_features).columns
        cats_names = self.transformers['cats'].fit_transform(self.categorical_features).columns #implement.named_transformers_['cats'].fit_transform(self.categorical_features).columns
        feat_names = np.concatenate([nums_names, cats_names])
        return feat_names

    # def fit(self, X):
    #     # self.dataset = X
    #     # self.numeric_features = X.select_dtypes(np.number)
    #     # self.categorical_features = X.select_dtypes(exclude=np.number)
    #     # self.implement = self.transform_columns()
    #     # self.transformed_data = self.implement.fit_transform(X)
    #     return self

    def transform(self, X):
        column_names = self.get_column_names()
        print("HELLO!!!IMHERE")
        return pd.DataFrame(super().transform(X).toarray(), columns=column_names, index=X.index)

        # column_names = self.get_column_names()
        # return pd.DataFrame(self.transformed_data, columns=column_names, index=X.index)





train_data = pd.read_csv('./data/marketing_train.csv')

a = TestSplit().split_test(train_data, "insurance_subscribe")
# print(a[0])

b = Preprocessor(a[0]).fit_transform(a[0])
print(b)

########################################

numeric_features = X_train.select_dtypes(np.number)
categorical_features = X_train.select_dtypes(exclude=np.number) #object, bool, category

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder()) #handle_unknown='ignore'
])

preprocessor = ColumnTransformer(
    transformers = [
        ('nums', numeric_transformer, numeric_features.columns),
        ('cats', categorical_transformer, categorical_features.columns)
    ] #n_jobs=-1
)

########################################

preprocessor_fit = preprocessor.fit_transform(X_train) ## -1~1  <- -44 999 1 -44 -1 scaler, ohe A a,b,c A_a, A_b, A_c
#preprocessor.tranform(test_data) ## a, b A_a, A_b   30 ~ 50 30 -1 50 1 -0.03 -

nums_names = preprocessor.named_transformers_['nums'].fit_transform(numeric_features).columns
cats_names = preprocessor.named_transformers_['cats'].fit_transform(categorical_features).columns
feat_names = np.concatenate([nums_names, cats_names])


preprocessed_train = pd.DataFrame(preprocessor_fit, columns=feat_names)

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])


model = LogisticRegression()

model.fit(preprocessed_train, y_train)
model.predict(preprocessed_train)
# print(preprocessed_train)

preprocessed_test = preprocessor.transform(X_test)
# print(preprocessed_test)
model.predict(preprocessed_test)

# print(model.score(preprocessed_train, y_train))
# print(model.score(preprocessed_test, y_test))

# print(pipe.fit_transform(X_train))

##2. STOP이 나오는 이유???!

test_data = pd.read_csv('./data/marketing_test.csv')



# pred.csv : ID, Predicted value
# report.csv : Precision, Recall, Accuracy, F1