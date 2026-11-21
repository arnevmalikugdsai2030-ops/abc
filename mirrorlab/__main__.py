"""
Package entrypoint for `python -m mirrorlab`.
"""

import sys

from mirrorlab import __version__


def main():
    print(f"MIRRORLAB Visual Task Factory v{__version__}")
    print("Commands:")
    print("  mirrorlab generate   - Generate synthetic benchmark tasks")
    print("  mirrorlab evaluate   - Evaluate a vision-language model")
    print("  mirrorlab app        - Launch Streamlit interactive visual studio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
