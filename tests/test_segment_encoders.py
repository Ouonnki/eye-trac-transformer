# -*- coding: utf-8 -*-
"""片段编码器单元测试"""

import unittest

import torch

from src.models.encoders import GazeCnnEncoder, GazeRnnEncoder


def build_mask(lengths, max_len):
    mask = torch.zeros((len(lengths), max_len), dtype=torch.bool)
    for i, length in enumerate(lengths):
        if length > 0:
            mask[i, :length] = True
    return mask


class SegmentEncoderShapeTest(unittest.TestCase):
    def test_cnn1d_output_shape(self):
        encoder = GazeCnnEncoder(
            input_dim=7,
            channels=[16, 16],
            kernel_sizes=[3, 3],
            dropout=0.0,
            output_dim=32,
        )
        x = torch.randn(2, 10, 7)
        mask = build_mask([10, 6], 10)
        output, _ = encoder(x, mask)
        self.assertEqual(tuple(output.shape), (2, 32))

    def test_rnn_variants_output_shape(self):
        cases = [
            ("rnn", False),
            ("lstm", False),
            ("gru", False),
            ("lstm", True),
        ]
        for rnn_type, bidirectional in cases:
            encoder = GazeRnnEncoder(
                rnn_type=rnn_type,
                input_dim=7,
                hidden_size=16,
                num_layers=1,
                dropout=0.0,
                bidirectional=bidirectional,
                output_dim=32,
            )
            x = torch.randn(2, 8, 7)
            mask = build_mask([8, 0], 8)
            output, _ = encoder(x, mask)
            self.assertEqual(tuple(output.shape), (2, 32))


if __name__ == "__main__":
    unittest.main()
