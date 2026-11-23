"""
Integration and Rendering Tests for MIRRORLAB UI Dashboard Views.

Verifies:
1. All 6 main platform views render without exceptions or missing data:
   - 🎯 Judge Demo Walkthrough
   - 🔬 Scene Laboratory
   - ⚔️ Model Arena
   - 📊 Scientific Analytics
   - 📈 Benchmark Dashboard
   - 🛠️ Physical Rig Guide
2. Streamlit AppTest integration: verifies clean mounting of the app and navigation.
3. Backend data loaders supply complete, typed statistics for all views.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from mirrorlab.ui.data_loader import (
    get_dataset_statistics,
    get_dataset_validation_status,
    get_recent_evaluation_runs,
    get_task_distribution,
)

APP_PATH = Path(__file__).parent.parent / "mirrorlab" / "ui" / "app.py"


class TestAppViews:
    """Verifies that all 6 dashboard views and data dependencies load cleanly."""

    def test_backend_data_loaders_integrity(self) -> None:
        """Verifies all UI data loader functions return valid structures."""
        stats = get_dataset_statistics()
        assert stats["has_data"] is True
        assert stats["total_tasks"] > 0
        assert "split_counts" in stats

        dist = get_task_distribution()
        assert "family_counts" in dist
        assert len(dist["family_counts"]) > 0

        runs = get_recent_evaluation_runs()
        assert isinstance(runs, list)

        val_status = get_dataset_validation_status()
        assert "checks" in val_status
        assert val_status["total_checks"] >= 10

    def test_apptest_default_walkthrough(self) -> None:
        """Verifies that the default view (Judge Demo Walkthrough) mounts with 0 exceptions."""
        at = AppTest.from_file(APP_PATH, default_timeout=20)
        at.run()
        assert len(at.exception) == 0, f"AppTest threw exceptions: {at.exception}"

    def test_apptest_physical_rig_guide(self) -> None:
        """Verifies that the Physical Rig Guide view mounts with 0 exceptions."""
        at = AppTest.from_file(APP_PATH, default_timeout=20)
        at.run()
        at.sidebar.radio[0].set_value("🛠️ Physical Rig Guide")
        at.run()
        assert len(at.exception) == 0, f"Rig guide threw exceptions: {at.exception}"

    def test_apptest_scene_laboratory(self) -> None:
        """Verifies that the Scene Laboratory view mounts with 0 exceptions."""
        at = AppTest.from_file(APP_PATH, default_timeout=20)
        at.run()
        at.sidebar.radio[0].set_value("🔬 Scene Laboratory")
        at.run()
        assert len(at.exception) == 0, f"Scene lab threw exceptions: {at.exception}"

    def test_apptest_model_arena(self) -> None:
        """Verifies that the Model Arena view mounts with 0 exceptions."""
        at = AppTest.from_file(APP_PATH, default_timeout=20)
        at.run()
        at.sidebar.radio[0].set_value("⚔️ Model Arena")
        at.run()
        assert len(at.exception) == 0, f"Model arena threw exceptions: {at.exception}"

    def test_apptest_scientific_analytics(self) -> None:
        """Verifies that the Scientific Analytics view mounts with 0 exceptions."""
        at = AppTest.from_file(APP_PATH, default_timeout=20)
        at.run()
        at.sidebar.radio[0].set_value("📊 Scientific Analytics")
        at.run()
        assert len(at.exception) == 0, f"Analytics threw exceptions: {at.exception}"

    def test_apptest_benchmark_dashboard(self) -> None:
        """Verifies that the Benchmark Dashboard view mounts with 0 exceptions."""
        at = AppTest.from_file(APP_PATH, default_timeout=20)
        at.run()
        at.sidebar.radio[0].set_value("📈 Benchmark Dashboard")
        at.run()
        assert len(at.exception) == 0, (
            f"Benchmark Dashboard threw exceptions: {at.exception}"
        )
