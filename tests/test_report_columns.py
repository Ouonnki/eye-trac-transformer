# -*- coding: utf-8 -*-
"""报表列一致性测试"""

import unittest

from scripts.build_task_embedding_report import METRIC_COLUMNS, build_single_rows


class ReportColumnsTest(unittest.TestCase):
    def test_single_report_columns(self):
        metrics = {
            "labels": [0, 1, 2, 1],
            "predictions": [0, 1, 2, 1],
            "probabilities": [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.2, 0.6, 0.2],
            ],
        }
        data = {"InDist": metrics}
        rows = build_single_rows(data, "cnn1d")
        self.assertEqual(len(rows), 1)
        expected_cols = ["分布", "编码器"] + METRIC_COLUMNS
        self.assertEqual(set(rows[0].keys()), set(expected_cols))


if __name__ == "__main__":
    unittest.main()
