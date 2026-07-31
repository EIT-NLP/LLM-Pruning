"""Small discovery CLI installed with the package."""

from __future__ import annotations

import argparse
import json

from . import list_kernels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List DynamicPruningKernels implementations"
    )
    parser.add_argument("family", nargs="?", help="Optional kernel family")
    parser.add_argument(
        "--available",
        action="store_true",
        help="Only show backends whose optional dependencies are importable",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    catalog = list_kernels(args.family, available_only=args.available)
    if args.json:
        print(json.dumps(catalog, indent=2))
        return

    for item in catalog:
        versions = ",".join(f"v{version}" for version in item["versions"]) or "-"
        architectures = ",".join(item["architectures"]) or "runtime"
        status = "available" if item["available"] else "missing deps"
        print(
            f"{item['family']:20} {item['backend']:10} "
            f"{versions:18} {architectures:12} {status}"
        )


if __name__ == "__main__":
    main()
