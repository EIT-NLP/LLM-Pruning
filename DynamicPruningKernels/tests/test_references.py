import unittest

import torch
import torch.nn.functional as F

from dynamic_width_jit import run_kernel


class TorchReferenceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.x = torch.randn(2, 3, 8)
        self.out_weight = torch.randn(12, 8)
        self.down_weight = torch.randn(5, 12)
        self.mask = torch.tensor(
            [
                [[True, False, True], [False, True, True], [True, True, False]],
                [[False, False, True], [True, False, False], [True, True, True]],
            ]
        )

    def test_gemm_mn_group_mask(self):
        output = run_kernel(
            "gemm_mn",
            self.x,
            self.out_weight,
            self.mask,
            backend="torch",
        )
        expected = self.x @ self.out_weight.T
        expected = expected.reshape(2, 3, 3, 4)
        expected = expected.masked_fill(~self.mask[..., None], 0).reshape(2, 3, 12)
        torch.testing.assert_close(output, expected)

    def test_gemm_k_group_mask(self):
        intermediate = self.x @ self.out_weight.T
        output = run_kernel(
            "gemm_k",
            intermediate,
            self.down_weight,
            self.mask,
            backend="torch",
        )
        masked = intermediate.reshape(2, 3, 3, 4)
        masked = masked.masked_fill(~self.mask[..., None], 0).reshape(2, 3, 12)
        torch.testing.assert_close(output, masked @ self.down_weight.T)

    def test_one_group_skips_whole_token(self):
        mask = torch.tensor([[True, False, True], [False, True, False]])
        output = run_kernel(
            "gemm_mn",
            self.x,
            self.out_weight,
            mask,
            backend="torch",
        )
        expected = (self.x @ self.out_weight.T) * mask[..., None]
        torch.testing.assert_close(output, expected)

    def test_attention_decode_leftpad(self):
        query = torch.randn(1, 1, 4, 8)
        key = torch.randn(1, 4, 2, 8)
        value = torch.randn(1, 4, 2, 8)
        route_mask = torch.ones(1, 1, 2, dtype=torch.bool)
        output = run_kernel(
            "attention_decode",
            query,
            key,
            value,
            route_mask,
            backend="torch",
            Leftpad=torch.tensor([2], dtype=torch.uint32),
            is_causal=False,
        )

        expected_valid = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key[:, 2:].transpose(1, 2),
            value[:, 2:].transpose(1, 2),
            is_causal=False,
            scale=8**-0.5,
            enable_gqa=True,
        ).transpose(1, 2)
        torch.testing.assert_close(output, expected_valid)


if __name__ == "__main__":
    unittest.main()
