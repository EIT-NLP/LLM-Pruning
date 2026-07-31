import unittest

import run as source_runner

from dynamic_width_jit import (
    KernelVariantError,
    get_kernel,
    list_kernels,
)
from dynamic_width_jit.catalog import KERNEL_CATALOG, discover_cute_variants


class RegistryTest(unittest.TestCase):
    def test_catalog_contains_all_families(self):
        families = {item["family"] for item in list_kernels()}
        self.assertEqual(
            families,
            {"gemm_mn", "gemm_k", "attention_prefill", "attention_decode"},
        )

    def test_family_defaults(self):
        defaults = {
            "gemm_mn": 1,
            "gemm_k": 1,
            "attention_prefill": 3,
            "attention_decode": 0,
        }
        for family, version in defaults.items():
            self.assertEqual(
                get_kernel(family, "cute", arch="sm12x").version, version
            )

    def test_latest_is_architecture_aware(self):
        gemm_mn = get_kernel(
            "gemm_mn",
            "cute",
            version="latest",
            arch="sm12x",
        )
        self.assertEqual(gemm_mn.version, 1)
        self.assertEqual(gemm_mn.arch, "sm12x")

        attention = get_kernel(
            "attention_prefill",
            "cute",
            version=-1,
            arch="sm12x",
        )
        self.assertEqual(attention.version, 3)

    def test_invalid_version_arch_pair_is_rejected(self):
        with self.assertRaises(KernelVariantError):
            get_kernel("gemm_mn", "cute", version=5, arch="sm12x")

    def test_aliases(self):
        kernel = get_kernel("gemm_k", "jit", version=1, arch="sm12x")
        self.assertEqual(kernel.backend, "cute")

    def test_cute_variants_follow_header_tree(self):
        for family in (
            "gemm_mn",
            "gemm_k",
            "attention_prefill",
            "attention_decode",
        ):
            registered = next(
                item
                for item in list_kernels(family)
                if item["backend"] == "cute"
            )
            expected = {
                f"v{version}": list(architectures)
                for version, architectures in discover_cute_variants(family).items()
            }
            self.assertEqual(registered["variants"], expected)

    def test_source_tree_and_installed_catalog_match(self):
        from python.catalog import KERNEL_CATALOG as source_catalog

        self.assertEqual(tuple(source_catalog), tuple(KERNEL_CATALOG))
        for name, definition in source_catalog.items():
            self.assertEqual(
                definition.legacy_default_version,
                KERNEL_CATALOG[name].legacy_default_version,
            )

    def test_legacy_runner_stays_source_tree_first(self):
        self.assertEqual(
            set(source_runner.KERNEL_REGISTRY),
            set(KERNEL_CATALOG),
        )
        for item in source_runner.KERNEL_REGISTRY.values():
            self.assertTrue(item["module"].startswith("python."))

    def test_runner_exposes_autotune_and_tracing_controls(self):
        parser = source_runner.build_parser()
        args = parser.parse_args(
            ["--autotune", "false", "--tracing", "true"]
        )
        common = source_runner.build_common_args(args)
        self.assertFalse(common.autotune)
        self.assertTrue(common.tracing)


if __name__ == "__main__":
    unittest.main()
