import config
import argparse

from unittest import TestCase, main
from unittest.mock import patch


class TestInput(TestCase):
    @patch(
        "argparse.ArgumentParser.parse_args", return_value=argparse.Namespace()
    )
    def test_input(self, mock_get):
        # self.assertEqual(config.args.train, mock_get.train)
        config.input = mock_get.input
        config.prediction = mock_get.prediction
        config.report = mock_get.report
        config.target = mock_get.target


if __name__ == "__main__":
    main(exit=False)
