import os
import unittest

import torch

from dynamic_width_jit import get_kernel


RUN_GPU = os.environ.get("DYNAMIC_WIDTH_RUN_GPU_TESTS") == "1"
RUN_TILELANG = os.environ.get("DYNAMIC_WIDTH_RUN_TILELANG_TESTS") == "1"


@unittest.skipUnless(torch.cuda.is_available() and RUN_GPU, "GPU smoke is opt-in")
class GPUBackendSmokeTest(unittest.TestCase):
    dtype = torch.float16
    device = "cuda"

    @staticmethod
    def assert_kernel_close(actual, expected):
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    def setUp(self):
        torch.manual_seed(0)

    def test_sm120a_jit_arch_is_persistent(self):
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("sm120a target is Blackwell-specific")
        from dynamic_width_jit.utils import _configure_sglang_jit_cuda_arch
        from sglang.jit_kernel import utils as jit_utils

        _configure_sglang_jit_cuda_arch("sm12x")
        self.assertEqual(jit_utils.get_jit_cuda_arch().target_name, "12.0a")

    def test_cute_gemm_mn(self):
        a = torch.randn(1, 16, 128, device=self.device, dtype=self.dtype)
        b = torch.randn(256, 128, device=self.device, dtype=self.dtype)
        mask = torch.rand(1, 16, 2, device=self.device) > 0.5
        expected = get_kernel("gemm_mn", "torch")(a, b, mask)
        actual, metadata = get_kernel("gemm_mn", "cute")(
            a, b, mask, autotune=False, estimate_sparsity=0.5
        )
        self.assertEqual(set(metadata), {"sorted_mask", "sorted_indices"})
        reused, _ = get_kernel("gemm_mn", "cute")(
            a,
            b,
            mask,
            sorted_mask=metadata["sorted_mask"],
            sorted_indices=metadata["sorted_indices"],
            autotune=False,
            estimate_sparsity=0.5,
        )
        self.assert_kernel_close(actual, expected)
        self.assert_kernel_close(reused, expected)

    def test_cute_gemm_k(self):
        a = torch.randn(1, 16, 256, device=self.device, dtype=self.dtype)
        b = torch.randn(128, 256, device=self.device, dtype=self.dtype)
        mask = torch.rand(1, 16, 2, device=self.device) > 0.5
        expected = get_kernel("gemm_k", "torch")(a, b, mask)
        actual, metadata = get_kernel("gemm_k", "cute")(
            a, b, mask, autotune=False, estimate_sparsity=0.5
        )
        self.assertEqual(set(metadata), {"sorted_mask", "sorted_indices"})
        reused, _ = get_kernel("gemm_k", "cute")(
            a,
            b,
            mask,
            sorted_mask=metadata["sorted_mask"],
            sorted_indices=metadata["sorted_indices"],
            autotune=False,
            estimate_sparsity=0.5,
        )
        self.assert_kernel_close(actual, expected)
        self.assert_kernel_close(reused, expected)

    def test_cute_gemv_metadata_is_empty(self):
        mn_input = torch.randn(
            1, 1, 256, device=self.device, dtype=self.dtype
        )
        mn_mask = torch.tensor(
            [[[True, False]]], device=self.device, dtype=torch.bool
        )
        mn_weight = torch.randn(256, 256, device=self.device, dtype=self.dtype)
        mn_expected = get_kernel("gemm_mn", "torch")(
            mn_input, mn_weight, mn_mask
        )
        mn_actual, mn_metadata = get_kernel("gemm_mn", "cute")(
            mn_input, mn_weight, mn_mask, autotune=False
        )

        k_input = torch.randn(
            1, 1, 512, device=self.device, dtype=self.dtype
        )
        k_mask = torch.tensor(
            [[[True, False]]], device=self.device, dtype=torch.bool
        )
        k_weight = torch.randn(128, 512, device=self.device, dtype=self.dtype)
        k_expected = get_kernel("gemm_k", "torch")(
            k_input, k_weight, k_mask
        )
        k_actual, k_metadata = get_kernel("gemm_k", "cute")(
            k_input, k_weight, k_mask, autotune=False
        )
        self.assertEqual(mn_metadata, {})
        self.assertEqual(k_metadata, {})
        self.assert_kernel_close(mn_actual, mn_expected)
        self.assert_kernel_close(k_actual, k_expected)

    def test_cute_gemm_k_small_m_dispatch_boundary(self):
        cases = [
            # M, K, G, expected GEMV dispatch
            (1, 256, 128, False),
            (2, 256, 128, True),
            (2, 256, 64, False),
            (4, 256, 64, True),
            (4, 128, 32, False),
            (1, 512, 256, True),
            (8, 1024, 512, False),
        ]
        for m, k, group_size, expects_gemv in cases:
            with self.subTest(M=m, K=k, G=group_size):
                a = torch.randn(
                    1, m, k, device=self.device, dtype=self.dtype
                )
                weight = torch.randn(
                    128, k, device=self.device, dtype=self.dtype
                )
                mask = (
                    torch.rand(
                        1,
                        m,
                        k // group_size,
                        device=self.device,
                    )
                    > 0.5
                )
                expected = get_kernel("gemm_k", "torch")(a, weight, mask)
                actual, metadata = get_kernel("gemm_k", "cute")(
                    a,
                    weight,
                    mask,
                    autotune=False,
                    estimate_sparsity=0.5,
                )
                if expects_gemv:
                    self.assertEqual(metadata, {})
                else:
                    self.assertEqual(
                        set(metadata), {"sorted_mask", "sorted_indices"}
                    )
                self.assert_kernel_close(actual, expected)

    def test_cute_gemm_mn_small_m_dispatch_boundary(self):
        weight = torch.randn(256, 128, device=self.device, dtype=self.dtype)

        gemv_input = torch.randn(1, 4, 128, device=self.device, dtype=self.dtype)
        gemv_mask = torch.rand(1, 4, 2, device=self.device) > 0.5
        gemv_expected = get_kernel("gemm_mn", "torch")(
            gemv_input, weight, gemv_mask
        )
        gemv_actual, gemv_metadata = get_kernel("gemm_mn", "cute")(
            gemv_input,
            weight,
            gemv_mask,
            autotune=False,
            estimate_sparsity=0.5,
        )
        self.assertEqual(gemv_metadata, {})
        self.assert_kernel_close(gemv_actual, gemv_expected)

        gemm_input = torch.randn(1, 8, 128, device=self.device, dtype=self.dtype)
        gemm_mask = torch.rand(1, 8, 2, device=self.device) > 0.5
        gemm_expected = get_kernel("gemm_mn", "torch")(
            gemm_input, weight, gemm_mask
        )
        gemm_actual, gemm_metadata = get_kernel("gemm_mn", "cute")(
            gemm_input,
            weight,
            gemm_mask,
            autotune=False,
            estimate_sparsity=0.5,
        )
        self.assertEqual(set(gemm_metadata), {"sorted_mask", "sorted_indices"})
        self.assert_kernel_close(gemm_actual, gemm_expected)

        wide_group_weight = torch.randn(
            1024, 128, device=self.device, dtype=self.dtype
        )
        wide_group_mask = torch.rand(1, 4, 2, device=self.device) > 0.5
        wide_group_expected = get_kernel("gemm_mn", "torch")(
            gemv_input, wide_group_weight, wide_group_mask
        )
        wide_group_actual, wide_group_metadata = get_kernel("gemm_mn", "cute")(
            gemv_input,
            wide_group_weight,
            wide_group_mask,
            autotune=False,
            estimate_sparsity=0.5,
        )
        self.assertEqual(wide_group_metadata, {})
        self.assert_kernel_close(wide_group_actual, wide_group_expected)

    def test_cute_attention_prefill(self):
        q = torch.randn(1, 16, 4, 64, device=self.device, dtype=self.dtype)
        k = torch.randn(1, 16, 2, 64, device=self.device, dtype=self.dtype)
        v = torch.randn(1, 16, 2, 64, device=self.device, dtype=self.dtype)
        mask = torch.rand(1, 16, 2, device=self.device) > 0.5
        expected = get_kernel("attention_prefill", "torch")(
            q, k, v, mask, is_causal=True
        )
        actual, metadata = get_kernel("attention_prefill", "cute")(
            q,
            k,
            v,
            mask,
            is_causal=True,
            autotune=False,
            estimate_sparsity=0.5,
        )
        self.assertEqual(set(metadata), {"sorted_mask", "sorted_indices"})
        reused, _ = get_kernel("attention_prefill", "cute")(
            q,
            k,
            v,
            mask,
            sorted_mask=metadata["sorted_mask"],
            sorted_indices=metadata["sorted_indices"],
            is_causal=True,
            autotune=False,
            estimate_sparsity=0.5,
        )
        reused_flat, _ = get_kernel("attention_prefill", "cute")(
            q,
            k,
            v,
            mask,
            sorted_mask=metadata["sorted_mask"].flatten(0, 1),
            sorted_indices=metadata["sorted_indices"].flatten(0, 1),
            is_causal=True,
            autotune=False,
            estimate_sparsity=0.5,
        )
        triton = get_kernel("attention_prefill", "triton")(
            q,
            k,
            v,
            mask,
            is_causal=True,
            estimate_sparsity=0.5,
        )
        self.assert_kernel_close(actual, expected)
        self.assert_kernel_close(reused, expected)
        self.assert_kernel_close(reused_flat, expected)
        self.assert_kernel_close(triton, expected)

        mn_input = torch.randn(
            1, 16, 128, device=self.device, dtype=self.dtype
        )
        mn_weight = torch.randn(
            256, 128, device=self.device, dtype=self.dtype
        )
        mn_expected = get_kernel("gemm_mn", "torch")(
            mn_input, mn_weight, mask
        )
        mn_actual, _ = get_kernel("gemm_mn", "cute")(
            mn_input,
            mn_weight,
            mask,
            sorted_mask=metadata["sorted_mask"],
            sorted_indices=metadata["sorted_indices"],
            autotune=False,
            estimate_sparsity=0.5,
        )
        self.assert_kernel_close(mn_actual, mn_expected)

        k_input = actual.flatten(2)
        k_weight = torch.randn(
            128, 256, device=self.device, dtype=self.dtype
        )
        k_expected = get_kernel("gemm_k", "torch")(
            k_input, k_weight, mask
        )
        k_actual, _ = get_kernel("gemm_k", "cute")(
            k_input,
            k_weight,
            mask,
            sorted_mask=metadata["sorted_mask"],
            sorted_indices=metadata["sorted_indices"],
            autotune=False,
            estimate_sparsity=0.5,
        )
        self.assert_kernel_close(k_actual, k_expected)

    @unittest.skipUnless(RUN_TILELANG, "TileLang smoke is separately opt-in")
    def test_tilelang_attention_decode(self):
        q = torch.randn(1, 1, 16, 128, device=self.device, dtype=self.dtype)
        k = torch.randn(1, 32, 8, 128, device=self.device, dtype=self.dtype)
        v = torch.randn(1, 32, 8, 128, device=self.device, dtype=self.dtype)
        mask = torch.tensor(
            [[[True, False, True, True, False, True, False, True]]],
            device=self.device,
        )
        expected = get_kernel("attention_decode", "torch")(
            q, k, v, mask, is_causal=False
        )
        actual = get_kernel("attention_decode", "tilelang")(
            q,
            k,
            v,
            mask,
            is_causal=False,
            autotune=False,
            estimate_sparsity=0.5,
        )
        self.assert_kernel_close(actual, expected)

    @unittest.skipUnless(RUN_TILELANG, "TileLang smoke is separately opt-in")
    def test_tilelang_attention_decode_per_head_groups(self):
        q = torch.randn(1, 1, 16, 128, device=self.device, dtype=self.dtype)
        k = torch.randn(1, 32, 8, 128, device=self.device, dtype=self.dtype)
        v = torch.randn(1, 32, 8, 128, device=self.device, dtype=self.dtype)
        mask = (torch.arange(16, device=self.device) % 3 != 1).reshape(1, 1, 16)
        leftpad = torch.tensor([3], device=self.device, dtype=torch.uint32)
        expected = get_kernel("attention_decode", "torch")(
            q,
            k,
            v,
            mask,
            Leftpad=leftpad,
            is_causal=False,
        )
        actual = get_kernel("attention_decode", "tilelang")(
            q,
            k,
            v,
            mask,
            Leftpad=leftpad,
            is_causal=False,
            autotune=False,
            estimate_sparsity=0.5,
        )
        self.assert_kernel_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
