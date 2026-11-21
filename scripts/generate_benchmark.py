"""
Benchmark Dataset Generation CLI Script for MIRRORLAB.
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic MIRRORLAB visual task benchmark."
    )
    parser.add_argument(
        "--count", type=int, default=100, help="Number of benchmark tasks to generate"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/sample",
        help="Output directory for generated dataset",
    )
    args = parser.parse_args()

    print(
        f"MIRRORLAB Dataset Generator: seed={args.seed}, count={args.count}, target={args.out_dir}"
    )
    print(
        "Scaffold CLI initialized. Generator logic will be activated in Milestone 4 & 5."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
