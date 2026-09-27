"""Smoke and behaviour tests for the Streamlit app.

Streamlit ships an AppTest harness that executes the real script headlessly, so
these assert the UI still runs and still renders its core output rather than
relying on a manual click-through.
"""

import os

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py"
)


@pytest.fixture(scope="module")
def app():
    if not os.path.exists(APP_PATH):
        pytest.skip("app.py not present.")
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    return at


class TestAppLoads:
    def test_script_runs_without_exceptions(self, app):
        assert not app.exception, [e.value for e in app.exception]

    def test_hero_and_tabs_render(self, app):
        assert len(app.tabs) == 3
        body = " ".join(m.value for m in app.markdown)
        assert "Titanic Survival Simulator" in body

    def test_simulator_form_controls_exist(self, app):
        assert len(app.selectbox) >= 4
        assert len(app.slider) >= 2
        assert len(app.button) >= 1

    def test_missing_model_shows_an_error_not_a_crash(self, monkeypatch):
        """With no artifact on disk the app must st.stop() cleanly."""
        at = AppTest.from_file(APP_PATH, default_timeout=180)
        real_exists = os.path.exists

        def fake_exists(path):
            if "best_model.pkl" in str(path):
                return False
            return real_exists(path)

        monkeypatch.setattr(os.path, "exists", fake_exists)
        at.run()
        assert not at.exception


class TestSimulationFlow:
    def test_clicking_the_button_produces_a_probability(self, app):
        at = AppTest.from_file(APP_PATH, default_timeout=180)
        at.run()
        at.button[0].click().run()
        assert not at.exception
        body = " ".join(m.value for m in at.markdown)
        assert "RESCUED" in body or "PERISHED" in body

    def test_selecting_a_female_first_class_passenger_survives(self, app):
        at = AppTest.from_file(APP_PATH, default_timeout=180)
        at.run()
        at.selectbox[0].set_value(1)      # Pclass -> 1st
        at.selectbox[1].set_value("female")
        at.selectbox[2].set_value("Mrs")  # Title -> Mrs
        at.button[0].click().run()
        assert not at.exception
        body = " ".join(m.value for m in at.markdown)
        assert "RESCUED" in body

    def test_selecting_a_male_third_class_passenger_perishes(self, app):
        at = AppTest.from_file(APP_PATH, default_timeout=180)
        at.run()
        at.selectbox[0].set_value(3)      # Pclass -> 3rd
        at.selectbox[1].set_value("male")
        at.selectbox[2].set_value("Mr")   # Title -> Mr
        at.button[0].click().run()
        assert not at.exception
        body = " ".join(m.value for m in at.markdown)
        assert "PERISHED" in body

    def test_blank_cabin_input_does_not_crash(self, app):
        at = AppTest.from_file(APP_PATH, default_timeout=180)
        at.run()
        at.text_input[0].set_value("").run()
        at.button[0].click().run()
        assert not at.exception

    def test_a_cabin_label_does_not_crash(self, app):
        at = AppTest.from_file(APP_PATH, default_timeout=180)
        at.run()
        at.text_input[0].set_value("C85").run()
        at.button[0].click().run()
        assert not at.exception


class TestAnalyticsTab:
    def test_benchmark_table_renders_when_artefacts_exist(self, app):
        at = AppTest.from_file(APP_PATH, default_timeout=180)
        at.run()
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.exists(os.path.join(repo_root, "reports", "cv_results.csv")):
            pytest.skip("Benchmark artefacts not generated yet.")
        assert len(at.dataframe) >= 1

    def test_spec_payload_renders_as_json(self, app):
        at = AppTest.from_file(APP_PATH, default_timeout=180)
        at.run()
        assert len(at.json) >= 1

    def test_spec_json_is_valid_json(self, app):
        import json as json_mod

        at = AppTest.from_file(APP_PATH, default_timeout=180)
        at.run()
        payload = json_mod.loads(at.json[0].value)
        assert payload["Dataset Folds"] == "5-Fold Stratified K-Fold"
        assert "Target Variable" in payload
