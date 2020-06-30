import argparse
import pandas as pd

from ml_class import TestSplit, Preprocessor, RobustScaler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def main():
    preprocessor = Preprocessor(RobustScaler())
    model = LogisticRegression(max_iter=600)

    train_score, validation_score = process_train_data(preprocessor, model)
    test_score = process_test_data(preprocessor, model)

    # for 

    save_report(train_score, validation_score, test_score)
    
def process_train_data(preprocessor, model):
    # get dataset
    train_data = get_data('marketing_train.csv')

    # split data
    target = get_arguments().target
    X_train, X_validation, y_train, y_validation = TestSplit().split_test(train_data, target)

    # fit & transform
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_validation_transformed = transform_data(preprocessor, X_validation)

    # fit & predict
    model.fit(X_train_transformed, y_train)
    X_train_prediction = predict_data(model, X_train_transformed)
    X_validatrion_prediction = predict_data(model, X_validation_transformed)

    # score
    train_score = get_scores(y_train, X_train_prediction)
    validation_score = get_scores(y_validation, X_validatrion_prediction)

    return train_score, validation_score

def process_test_data(preprocessor, model):
    # get dataset
    arguments = get_arguments()
    test_data = get_data(arguments.input)
    target = arguments.target

    # transform
    test_transformed = transform_data(preprocessor, test_data.drop(target, axis=1)) ##preprocessor

    # predict
    test_prediction = predict_data(model, test_transformed) ##model
    save_prediction(test_prediction, arguments.prediction)

    # score
    test_score = get_scores(test_data[target], test_prediction)

    return test_score

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='marketing_test.csv', help='Write file name you want to predict')
    parser.add_argument('--prediction', default='pred.csv', help='Write file name you want to save prediction')
    parser.add_argument('--report', default='report.csv', help='Write file name you want to save report')
    parser.add_argument('--target', default='insurance_subscribe', help='Write target feature')

    return parser.parse_args()

def get_data(file_name):
    return pd.read_csv('./data/{}'.format(file_name))

def transform_data(preprocessor, dataset_without_target):
    return preprocessor.transform(dataset_without_target)

def predict_data(model, transformed_data):
    return model.predict(transformed_data)

def get_scores(actual_y, predicted_y):
    funcs = [precision_score, recall_score, accuracy_score, f1_score]
    scores = []

    for func in funcs:
        scores.append(func(actual_y, predicted_y))

    return scores

def save_prediction(prediction, file_name):
    prediction_df = pd.DataFrame(prediction, columns=["Predicted value"]).rename_axis("ID")
    prediction_df.to_csv('./result/{}'.format(file_name))

    return prediction_df

def save_report(train_score, validation_score, test_score):
    index_list = [
        ['Robust-Logistic', 'Robust-Logistic', 'Robust-Logistic'],
        ['train', 'validation', 'test']
    ]
    index = pd.MultiIndex.from_arrays(index_list, names=('scaler_model', 'data'))

    report_df = pd.DataFrame(index=index, columns=["Precision", "Recall", "Accuracy", "F1"])

    dataset_type = {'train': train_score, 'validation': validation_score, 'test': test_score}
    
    for data, score in dataset_type.items():
        report_df.loc[('Robust-Logistic', data)] = score
    
    report_df.to_csv('./result/{}'.format(get_arguments().report))

    print(report_df)

    return report_df

if __name__ == "__main__":
    main()

# python ml_data.py --input “marketing_test.csv” --prediction “pred.csv” --report “report.csv”