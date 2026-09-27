"""Tests for the gauge and verdict rendering helpers."""

import pytest

from explain import render_gauge, verdict_class


class TestRenderGauge:
    def test_contains_the_dial_structure(self):
        html = render_gauge(0.5)
        for part in ("gauge-wrap", "gauge-track", "gauge-needle", "gauge-pin", "gauge-readout"):
            assert part in html

    def test_needle_angle_is_a_full_sweep_across_the_range(self):
        assert "rotate(0.0deg)" in render_gauge(0.0)
        assert "rotate(90.0deg)" in render_gauge(0.5)
        assert "rotate(180.0deg)" in render_gauge(1.0)

    def test_probability_is_shown_as_a_percentage(self):
        assert "0.5%" in render_gauge(0.005)
        assert "50.0%" in render_gauge(0.5)
        assert "100.0%" in render_gauge(1.0)

    @pytest.mark.parametrize("bad", [-0.5, -0.01, 1.5, 42.0])
    def test_out_of_range_input_is_clamped(self, bad):
        html = render_gauge(bad)
        assert "0.0%" in html or "100.0%" in html

    def test_clamping_keeps_the_needle_inside_the_dial(self):
        assert "rotate(0.0deg)" in render_gauge(-3.0)
        assert "rotate(180.0deg)" in render_gauge(9.0)

    def test_accepts_numpy_floats(self):
        import numpy as np

        assert "rotate(90.0deg)" in render_gauge(np.float64(0.5))

    def test_no_unresolved_placeholder_braces(self):
        assert "{" not in render_gauge(0.42)
        assert "}" not in render_gauge(0.42)


class TestVerdictClass:
    def test_threshold_boundary_is_safe(self):
        assert verdict_class(0.5) == "verdict-safe"
        assert verdict_class(1.0) == "verdict-safe"

    def test_below_threshold_is_risky(self):
        assert verdict_class(0.49) == "verdict-risky"
        assert verdict_class(0.0) == "verdict-risky"

    def test_custom_threshold_is_honoured(self):
        assert verdict_class(0.3, threshold=0.2) == "verdict-safe"
        assert verdict_class(0.9, threshold=0.95) == "verdict-risky"


class TestAccessibilityPolish:
    def test_stylesheet_respects_reduced_motion(self, repo_root):
        import os

        with open(os.path.join(repo_root, "app.py"), encoding="utf-8") as fh:
            source = fh.read()
        assert "prefers-reduced-motion" in source
        assert "prefers-color-scheme" in source
