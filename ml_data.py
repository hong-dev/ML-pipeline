import pandas as pd
import numpy as np

from sklearn.compose import make_column_transformer, ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import RobustScaler as SklearnRobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer as SklearnSimpleImputer
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from sklearn.compose import ColumnTransformer as SklearnColumnTransformer

class ColumnTransformer(SklearnColumnTransformer):
    # def transform(self, X) -> pd.DataFrame:
    #     return pd.DataFrame(super().transform(X), columns=X.columns, index=X.index)

    # def fit_transform(self, X):
    #     # X = self.named_transformers_
    #     return pd.DataFrame(super().fit_transform(X), columns=X.columns, index=X.index)

    def merge(self, X, y=None):
        return pd.DataFrame(super().fit_transform(X), columns=X.columns, index=X.index)

    def transform(self, X):
        sparse_matrix = super().fit_transform(X)
        a = self.transformers[0][1]
        print(a)
        print(pd.DataFrame(a).columns)
        return pd.DataFrame(sparse_matrix.toarray(), columns=a.columns, index=X.index)

    def fit_transform(self, X, y=None):
        # self.fit(X)
        return self.transform(X)

class RobustScaler(SklearnRobustScaler):
    def fit_transform(self, X, y=None):
        return pd.DataFrame(super().fit_transform(X), columns=X.columns, index=X.index)

class SimpleImputer(SklearnSimpleImputer):
    def fit_transform(self, X, y=None):
        return pd.DataFrame(super().fit_transform(X), columns=X.columns, index=X.index)

class OneHotEncoder(SklearnOneHotEncoder):
    def transform(self, X):
        sparse_matrix = super(OneHotEncoder, self).transform(X)
        new_columns = self.get_new_columns(X=X)
        return pd.DataFrame(sparse_matrix.toarray(), columns=new_columns, index=X.index)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_new_columns(self, X):
        new_columns = []
        for i, column in enumerate(X.columns):
            j = 0
            while j < len(self.categories_[i]):
                new_columns.append(f'{column}_{self.categories_[i][j]}')
                j += 1
        return new_columns


train_data = pd.read_csv('./data/marketing_train.csv')
test_data = pd.read_csv('./data/marketing_test.csv')

# target = 'insurance_subscribe'

# X = train_data.drop(target, axis=1)
# y = train_data[target]

# X_train, X_test, y_train, y_test = train_test_split(X, y)

# onehot_marketing = OneHotEncoder().fit_transform(marketing.select_dtypes(include=['object']))

# onehot_data = pd.DataFrame(onehot_marketing.toarray())


# process = make_column_transformer(
#     (OneHotEncoder(categories='auto'), [0]),
#     remainder='passthrough'
# )


numeric_features = train_data.select_dtypes(np.number).columns
categorical_features = train_data.select_dtypes(['object', 'bool', 'category']).columns

# numeric_features = train_data.select_dtypes(exclude=['object']).columns
# categorical_features = train_data.select_dtypes(include=['object']).columns

# numeric_features = make_column_selector(marketing, dtype_exclude='object')
# categorical_features = make_column_selector(marketing, dtype_include='object')
##TypeError: 'DataFrame' objects are mutable, thus they cannot be hashed

# numeric_features = list(train_data.columns[train_data.dtypes == 'int64'])
# categorical_features = list(train_data.columns[train_data.dtypes == 'object'])



numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder()), #handle_unknown='ignore'
])


preprocessor = ColumnTransformer(
    transformers = [
        ('nums', numeric_transformer, numeric_features),
        ('cats', categorical_transformer, categorical_features)
    ] #n_jobs=-1
)

## ValueError: No valid specification of the columns. Only a scalar, list or slice of all integers or all strings, or boolean mask is allowed
## df.select_dtypes() doesn't output column indices. It outputs a subset of the DataFrame with the matched columns.

# print(marketing.select_dtypes(include=['object']))
# print(make_column_selector(marketing, dtype_include='object'))


# print(OneHotEncoder().fit_transform(train_data.select_dtypes(['object'])))
# print(SimpleImputer(strategy='most_frequent').fit_transform(train_data.select_dtypes(['object'])))
# print(categorical_transformer.fit_transform(train_data.select_dtypes(['object'])))

# print(SimpleImputer(strategy='median').fit_transform(train_data.select_dtypes(np.number)))
# print(RobustScaler().fit_transform(train_data.select_dtypes(np.number)))
# print(numeric_transformer.fit_transform(train_data.select_dtypes(np.number)))


preprocessor_fit = preprocessor.fit_transform(train_data)
p = preprocessor.fit_transform(train_data)
# print(preprocessor.named_transformers_['cats'].fit_transform(train_data.select_dtypes(['object'])))

print(pd.DataFrame(p))

#### get column names
# cats_names = preprocessor.named_transformers_['cats']['onehot'].get_feature_names(categorical_features) #순서가 어찌되는지 찾아보기!
# cats_names = preprocessor.fit_transform(train_data).get_feature_names() #순서가 어찌되는지 찾아보기!
# print(cats_names)

# # pre2 = preprocessor.named_transformers_['nums']['scaler'].get_feature_names()
# # AttributeError: 'StandardScaler' object has no attribute 'get_feature_names'

# labels = np.concatenate([numeric_features, cats_names]) #순서가 어찌되는지 찾아보기!
# print(labels)

# print(pd.DataFrame(preprocessor_fit, columns = labels))



pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('claasifier', LogisticRegression(solver='lbfgs')) ##classifier
])

# pipe.fit(X, y)


# print(numeric_transformer.fit_transform(marketing.select_dtypes(exclude=['object'])))
# print(categorical_transformer.fit_transform(marketing.select_dtypes(include=['object'])).toarray())


#cross_val_score(pipe, x, y, cv=5, scoring='accuracy').mean()
#grid = GridSearchCV(model_pipeline, param_grid=param, cv=5).fit(X_train, y_train)
#grid.score(X_test, y_test)


# print(marketing.dtypes)
# print(marketing.isna().sum())
# print(marketing_test.isnull().sum())