# Overview

- Train the original train dataset (marketing_train.csv) to fit target feature (insurance_subscribe)
- Predict the test dataset (marketing_test.csv) and save to a csv file (pred.csv)
- Score the test dataset and save to a csv file (report.csv)

<br>

---

# Stack/Library

### python

- source codes are written in python

### pandas

- save prediction and score data in dataframe
- transform function in Preprocessor returns dataframe

### sklearn

- scaler, encoder, imputer for preprocessing data
- model, metrics for training and scoring data

### numpy

- concatenate column names of numeric and categorical features (numpy.concatenate)
- select numeric features in dataframe (numpy.number)

<br>

---

# Directory/File

- `data/` : includes original train set and test set
- `result/` : includes created prediction file and report(scores) file
- `ml_class.py` : classes for importing (TestSplit, Preprocessor)
- `ml_data.py` : main functions for data process

<br>

---

# How to run

```bash
# bash command
$ python ml_data.py
```

### Arguments

- `--train` : train dataset (default='marketing_train.csv')
- `--input` : test dataset (default='marketing_test.csv')
- `--prediction` : prediction file (default='pred.csv')
- `--report` : report file (default='report.csv')
- `--target` : target feature (default='insurance_subscribe')

<br>

---

# Process

- used `scalers` : StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
- used `models` : LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, LinearSVC

### 1. get dataset
- original train dataset and test dataset (two csv files)
### 2. split dataset
- split train dataset into train set and validation set

    ⇒ total three datasets : train set(split), validation set(split), test set(original)

### 3. preprocess
- numeric features : imputer, scaler
- categorical features : imputer, onehotencoder
### 4. predict
- save prediction for test set to a csv file
### 5. score
- accuracy_score, recall_score, precision_score, f1_score
- save scores for all dataset to a csv file

<br>

---

# Conclusion

### result/pred.csv

### result/report.csv