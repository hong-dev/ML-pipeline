import pandas as pd
import numpy as np

from sklearn.compose import make_column_transformer, ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer

train_data = pd.read_csv('./data/marketing_train.csv')
test_data = pd.read_csv('./data/marketing_test.csv')

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
    ('scaler', StandardScaler())

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

preprocessor_fit = preprocessor.fit_transform(train_data)

pre = preprocessor.named_transformers_['cats']['onehot'].get_feature_names(categorical_features)

# pre2 = preprocessor.named_transformers_['nums']['scaler'].get_feature_names()
# AttributeError: 'StandardScaler' object has no attribute 'get_feature_names'

labels = np.concatenate([numeric_features, pre])
print(labels)

print(pd.DataFrame(preprocessor_fit, columns = labels))


# print(numeric_transformer.fit_transform(train_data.select_dtypes(np.number)))
# print(categorical_transformer.fit_transform(train_data.select_dtypes(['object', 'bool', 'category'])))

# pipe = Pipeline([
#     ('preprocessor', preprocessor),
#     ('regression', LogisticRegression) ##classifier
# ])

# pipe.fit(train_data, train_data['insurance_subscribe'])


# print(numeric_transformer.fit_transform(marketing.select_dtypes(exclude=['object'])))
# print(categorical_transformer.fit_transform(marketing.select_dtypes(include=['object'])).toarray())


#cross_val_score(pipe, x, y, cv=5, scoring='accuracy').mean()
#grid = GridSearchCV(model_pipeline, param_grid=param, cv=5).fit(X_train, y_train)
#grid.score(X_test, y_test)


# print(marketing.dtypes)
# print(marketing.isna().sum())
# print(marketing_test.isnull().sum())