import unittest
import tempfile
import pandas as pd

from unittest.mock import patch
from itertools import product

from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    MaxAbsScaler,
)

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from main import main, split_test

# import config

### fit된 pipe가 진짜로 제대로 fit 되었는지 (pipe와 load로 비교)
### 

class TestMain(unittest.TestCase):
    # def setUp(self):
    #     self.test_dir = tempfile.TemporaryFile()

    # def tearDown(self):
    #     self.test_dir.close()

    # def _side_effect(self, data_name):
    #     inputs = {
    #         "titanic": {
    #         }
    #     }
    #     return inputs[data_name]


    def _run_main(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            print(tmpdirname)

            # main()
        
            self.assertEqual(1, 2)

    # # 내가 여기에서 mock을 해야하는 것은, config.py의 args?
    # @patch('config.train_file_path', return_value='AAAAAA', side_effect=["A", "B", "C"])
    # def test_titanic(self, mock_get):
    #     """
    #     dataset: actual csv data (titanic)
    #     """

    #     mock = mock_get.return_value

    #     train_data = pd.read_csv("./data/source_kaggle_titanic_train.csv")
    #     test_data = pd.read_csv("./data/source_kaggle_titanic_test_gt.csv")

    #     print(mock)
    #     print(mock_get)
    #     print(mock_get.side_effect)

    #     # self._run_main()
    
    def test_marketing(self):
        self._run_main()


class TestSplit(unittest.TestCase):
    def _split(self, dataset, target):
        X_train, X_validation, y_train, y_validation = split_test(dataset, target)
    
        self.assertEqual(len(X_train) + len(X_validation), len(dataset))
        self.assertEqual(len(y_train) + len(y_validation), len(dataset))
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_validation), len(y_validation))
        self.assertEqual(y_train.name, y_validation.name)
        self.assertEqual(X_train.columns.all(), X_validation.columns.all())
    
    def test_small_data(self):
        """
        dataset: two numeric features, two categorical features
        """

        data = {
            "num1": [1, 2, 3],
            "cat1": ["a", "b", "c"],
            "num2": [4, 5, 6],
            "cat2": ["d", "e", "f"],
            "target": [0, 1, 1]
        }
        dataset = pd.DataFrame(data)
        target = "target"

        self._split(dataset, target)

    def test_titanic(self):
        """
        dataset: actual csv data (titanic)
        """

        dataset = pd.read_csv("./data/source_kaggle_titanic_train.csv")
        target = "Survived"

        self._split(dataset, target)

    def test_marketing(self):
        """
        dataset: actual csv data (marketing)
        """

        dataset = pd.read_csv("./data/marketing_train.csv")
        target = "insurance_subscribe"

        self._split(dataset, target)


# class TestGetScores(unittest.TestCase):
#     def setUp(self):
#         self.scalers = [
#             StandardScaler,
#             MinMaxScaler,
#             RobustScaler,
#             MaxAbsScaler,
#         ]

#         self.models = [
#             (LogisticRegression, {"max_iter": 700}),
#             (DecisionTreeClassifier, {"max_depth": 4}),
#             (RandomForestClassifier, {"max_depth": 4}),
#             (LinearSVC, {"dual": False}),
#         ]

#     def _get_scores(self, actual_y, predicted_y):

#         for scaler, (model_class, params) in product(self.scalers, self.models):
#             get_scores(actual_y, predicted_y, scaler, model_class(**params), "train")

    
#     def test_titanic(self):
#         """
#         dataset: actual csv data (titanic)
#         """

#         dataset = pd.read_csv("./data/source_kaggle_titanic_train.csv")
#         target = "Survived"

#         actual_y = dataset[target]
#         predicted_y = pd.read_csv("./result/pred.csv")       

#         self._get_scores(actual_y, predicted_y)


if __name__ == "__main__":
    unittest.main(exit=False)