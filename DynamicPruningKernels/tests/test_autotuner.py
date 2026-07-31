import tempfile
import unittest
from unittest.mock import patch

from python.autotune import autotune
from python.autotune import autotuner as autotuner_module


class FakeModule:
    def __init__(self, block, calls):
        self.block = block
        self.calls = calls

    def run(self, *args):
        self.calls.append((self.block, args))


class RuntimeConfigAutotuneTest(unittest.TestCase):
    def test_runtime_configs_share_compiles_and_append_to_run(self):
        compile_calls = []
        run_calls = []
        configs = [
            {"BLOCK": block, "Split": split}
            for block in (32, 64)
            for split in (1, 2)
        ]

        with tempfile.TemporaryDirectory() as cache_dir:
            kernel_id = f"runtime_config_test_{id(self)}"

            @autotune(
                kernel_id=kernel_id,
                config_params=["BLOCK", "Split"],
                runtime_config_params=["Split"],
                configs=configs,
                key=["problem_size"],
                runtime_params=["payload"],
                cache_dir=cache_dir,
                compile_workers=2,
            )
            def compile_kernel(problem_size, BLOCK):
                compile_calls.append(BLOCK)
                return FakeModule(BLOCK, run_calls)

            def fake_do_bench(fn, **_kwargs):
                fn()
                block, args = run_calls[-1]
                split = args[-1]
                return float(abs(block - 64) + abs(split - 2))

            with patch.object(autotuner_module, "do_bench", side_effect=fake_do_bench):
                module = compile_kernel(problem_size=128, payload="tune")

            self.assertCountEqual(compile_calls, [32, 64])
            self.assertEqual(
                {(block, args[-1]) for block, args in run_calls},
                {(32, 1), (32, 2), (64, 1), (64, 2)},
            )
            self.assertTrue(all(args[:-1] == ("tune",) for _, args in run_calls))
            self.assertEqual(module.best_config.kwargs, {"BLOCK": 64, "Split": 2})

            run_calls.clear()
            module.run("launch")
            self.assertEqual(run_calls, [(64, ("launch", 2))])

            hot_module = compile_kernel(problem_size=128, payload="unused")
            self.assertIs(hot_module, module)
            self.assertCountEqual(compile_calls, [32, 64])

            cached_compile_calls = []
            cached_run_calls = []

            @autotune(
                kernel_id=kernel_id,
                config_params=["BLOCK", "Split"],
                runtime_config_params=["Split"],
                configs=configs,
                key=["problem_size"],
                runtime_params=["payload"],
                cache_dir=cache_dir,
                compile_workers=2,
            )
            def compile_cached_kernel(problem_size, BLOCK):
                cached_compile_calls.append(BLOCK)
                return FakeModule(BLOCK, cached_run_calls)

            with patch.object(
                autotuner_module,
                "do_bench",
                side_effect=AssertionError("cache hit must not benchmark"),
            ):
                cached_module = compile_cached_kernel(
                    problem_size=128,
                    payload="unused",
                )

            self.assertEqual(cached_compile_calls, [64])
            cached_module.run("cached")
            self.assertEqual(cached_run_calls, [(64, ("cached", 2))])

    def test_heuristic_runtime_config_is_bound_when_autotune_is_disabled(self):
        run_calls = []

        def heuristic(problem_size):
            return {"BLOCK": 32, "Split": 3, "Mode": 5}

        @autotune(
            kernel_id=f"runtime_config_heuristic_test_{id(self)}",
            config_params=["BLOCK", "Split", "Mode"],
            runtime_config_params=["Split", "Mode"],
            configs=[{"BLOCK": 32, "Split": 1, "Mode": 4}],
            key=["problem_size"],
            runtime_params=["payload"],
            heuristic=heuristic,
        )
        def compile_kernel(problem_size, BLOCK):
            return FakeModule(BLOCK, run_calls)

        module = compile_kernel(
            problem_size=128,
            payload="unused",
            autotune=False,
        )
        module.run("launch")

        self.assertEqual(
            module.best_config.kwargs,
            {"BLOCK": 32, "Split": 3, "Mode": 5},
        )
        self.assertEqual(run_calls, [(32, ("launch", 3, 5))])

    def test_runtime_config_must_be_an_autotune_config(self):
        with self.assertRaisesRegex(ValueError, "must be included in config_params"):
            @autotune(
                kernel_id="invalid_runtime_config_test",
                config_params=["BLOCK"],
                runtime_config_params=["Split"],
            )
            def compile_kernel(BLOCK):
                return FakeModule(BLOCK, [])


if __name__ == "__main__":
    unittest.main()
