"""
Package entrypoint for `python -m mirrorlab`.
"""

import subprocess
import sys
from pathlib import Path

from mirrorlab import __version__


def launch_app(extra_args: list[str] | None = None) -> int:
    """Launch the Streamlit interactive web studio."""
    app_file = Path(__file__).parent / "ui" / "app.py"
    if not app_file.exists():
        print(f"Error: UI app file not found at {app_file}")
        return 1

    cmd = [sys.executable, "-m", "streamlit", "run", str(app_file)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"Launching MIRRORLAB Optical Studio v{__version__}...")
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 0


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint accepting optional argument list."""
    args = argv if argv is not None else sys.argv[1:]
    if not args or args[0] in ("-h", "--help", "help") or args[0].startswith("-"):
        print(f"MIRRORLAB Visual Task Factory v{__version__}")
        print("Commands:")
        print("  mirrorlab app        - Launch Streamlit interactive visual studio")
        print("  mirrorlab audit      - Run 10-point dataset integrity audit")
        print("  mirrorlab generate   - Generate synthetic benchmark tasks")
        print("  mirrorlab evaluate   - Evaluate a vision-language model")
        return 0

    command = args[0].lower()
    if command == "app":
        return launch_app(args[1:])
    elif command == "audit":
        from mirrorlab.evaluation.dataset_auditor import DatasetAuditor

        print("Executing MIRRORLAB 10-point benchmark dataset audit...")
        auditor = DatasetAuditor()
        report = auditor.audit_all()
        print(f"Audit completed: {report.total_scenes_audited} scenes audited.")
        print(f"Integrity Status: {'PASSED' if report.is_dataset_valid else 'FAILED'}")
        return 0 if report.is_dataset_valid else 1
    else:
        print(f"Unknown command: '{command}'")
        print("Available commands: app, audit, generate, evaluate")
        return 1


if __name__ == "__main__":
    sys.exit(main())
