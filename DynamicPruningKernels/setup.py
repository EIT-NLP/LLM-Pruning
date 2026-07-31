"""Setuptools hooks for preserving the source tree while shipping JIT headers."""

from pathlib import Path
from shutil import copytree

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


REPOSITORY_ROOT = Path(__file__).parent
SOURCE_PACKAGE = REPOSITORY_ROOT / "python"


def discover_packages() -> list[str]:
    """Map every source ``python/**/__init__.py`` package into the wheel."""

    packages = ["dynamic_width_jit"]
    for init_file in sorted(SOURCE_PACKAGE.rglob("__init__.py")):
        relative = init_file.parent.relative_to(SOURCE_PACKAGE)
        if relative.parts:
            packages.append("dynamic_width_jit." + ".".join(relative.parts))
    return packages


class build_py(_build_py):
    """Copy runtime compilation inputs beside the installed Python package."""

    def run(self):
        super().run()
        package_root = Path(self.build_lib) / "dynamic_width_jit"
        for directory in ("include", "3rdparty", "csrc"):
            source = REPOSITORY_ROOT / directory
            if source.is_dir():
                copytree(
                    source,
                    package_root / directory,
                    dirs_exist_ok=True,
                )


setup(
    packages=discover_packages(),
    package_dir={"dynamic_width_jit": "python"},
    cmdclass={"build_py": build_py},
)
