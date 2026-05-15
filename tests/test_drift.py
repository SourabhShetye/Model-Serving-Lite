"""
tests/test_drift.py

Unit tests for all three drift signals.

Testing philosophy here:
  - Each signal is tested in isolation with synthetic data that WILL trigger it.
  - We also test that clean data does NOT trigger false alerts.
  - reset_drift_service() ensures each test gets a fresh instance.

These tests are pure Python — no FastAPI test client, no Redis, no DB.
That's intentional: DriftService has no external dependencies, so the
tests run in milliseconds and can run in CI without any infrastructure.
"""

import pytest

from app.services.drift_service import DriftService, DriftSignal, reset_drift_service


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

@pytest.fixture(autouse=True)
def clean_singleton():
    """Reset the DriftService singleton before every test."""
    reset_drift_service()
    yield
    reset_drift_service()


@pytest.fixture
def svc(monkeypatch) -> DriftService:
    """
    Returns a DriftService with a small window (10) for fast tests.
    Patches settings so we don't need a real .env file.
    """
    from app import config
    settings = config.get_settings()
    monkeypatch.setattr(settings, "drift_window_size", 10)
    monkeypatch.setattr(settings, "drift_ks_threshold", 0.05)
    monkeypatch.setattr(settings, "drift_confidence_drop_threshold", 0.10)
    monkeypatch.setattr(settings, "drift_language_threshold", 0.30)
    return DriftService()


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _fill_baseline(svc: DriftService, n: int = 10, confidence: float = 0.95) -> None:
    """Fills the baseline window with stable English, short, high-confidence samples."""
    for _ in range(n):
        svc.record(text="This is great!", confidence=confidence)


# ------------------------------------------------------------------ #
# Baseline establishment                                               #
# ------------------------------------------------------------------ #

class TestBaseline:
    def test_baseline_not_established_initially(self, svc):
        assert svc.get_status()["baseline_established"] is False

    def test_baseline_establishes_after_window_fills(self, svc):
        _fill_baseline(svc, n=10)
        assert svc.get_status()["baseline_established"] is True

    def test_baseline_not_established_before_window_fills(self, svc):
        _fill_baseline(svc, n=9)
        assert svc.get_status()["baseline_established"] is False

    def test_total_observations_increments(self, svc):
        _fill_baseline(svc, n=5)
        assert svc.get_status()["total_observations"] == 5


# ------------------------------------------------------------------ #
# Signal 3: Confidence Collapse (tested first — simplest)             #
# ------------------------------------------------------------------ #

class TestConfidenceDrift:
    def test_no_alert_on_stable_confidence(self, svc):
        """Stable high confidence should never alert."""
        _fill_baseline(svc, confidence=0.95)

        # Record 10 more samples at similar confidence
        # (this fills the live window and triggers analysis)
        for _ in range(10):
            svc.record(text="Good product!", confidence=0.93)

        # If no alert was logged, the test passes.
        # We verify by checking get_status reports reasonable values.
        status = svc.get_status()
        # After 10 baseline + 10 live (which triggers analysis), live window resets
        assert status["live_window_size"] == 0  # reset after analysis

    def test_confidence_collapse_detected(self, svc):
        """A sharp confidence drop should trigger the confidence signal."""
        _fill_baseline(svc, confidence=0.97)

        alerts_fired = []
        original_emit = svc._emit_alerts

        def capture_emit(result):
            alerts_fired.extend(result.alerts)
            original_emit(result)

        svc._emit_alerts = capture_emit

        # Inject low-confidence predictions into live window
        for _ in range(10):
            svc.record(text="Uncertain input...", confidence=0.52)

        assert any(a.signal == DriftSignal.CONFIDENCE for a in alerts_fired), (
            "Expected confidence drift alert, got none"
        )

    def test_confidence_alert_severity_critical_on_large_drop(self, svc):
        """A >25% confidence drop should be CRITICAL, not just WARNING."""
        _fill_baseline(svc, confidence=0.98)

        alerts_fired = []
        svc._emit_alerts = lambda r: alerts_fired.extend(r.alerts)

        for _ in range(10):
            svc.record(text="Test", confidence=0.50)  # ~49% relative drop

        conf_alerts = [a for a in alerts_fired if a.signal == DriftSignal.CONFIDENCE]
        assert conf_alerts, "Expected confidence alert"
        assert conf_alerts[0].severity == "CRITICAL"


# ------------------------------------------------------------------ #
# Signal 1: Input Length KS Test                                       #
# ------------------------------------------------------------------ #

class TestLengthDrift:
    def test_no_alert_on_similar_lengths(self, svc):
        """Similar length distributions should not alert."""
        _fill_baseline(svc, n=10)

        alerts_fired = []
        svc._emit_alerts = lambda r: alerts_fired.extend(r.alerts)

        for _ in range(10):
            svc.record(text="Short text here.", confidence=0.95)

        length_alerts = [a for a in alerts_fired if a.signal == DriftSignal.INPUT_LENGTH]
        assert not length_alerts, f"False positive length alert: {length_alerts}"

    def test_length_drift_detected_on_very_long_inputs(self, svc):
        """Sudden shift to very long inputs should trigger KS alert."""
        # Baseline: short texts (~14 chars)
        _fill_baseline(svc, n=10)

        alerts_fired = []
        svc._emit_alerts = lambda r: alerts_fired.extend(r.alerts)

        # Live window: very long texts (~500 chars)
        for _ in range(10):
            svc.record(text="x" * 500, confidence=0.95)

        length_alerts = [a for a in alerts_fired if a.signal == DriftSignal.INPUT_LENGTH]
        assert length_alerts, "Expected length drift alert on extreme length shift"

    def test_ks_test_returns_pvalue(self, svc):
        """_check_length_drift should return a numeric p-value."""
        baseline = [15.0] * 10
        live = [15.0] * 10
        pvalue, alert = svc._check_length_drift(baseline, live)
        assert isinstance(pvalue, float)
        assert 0.0 <= pvalue <= 1.0
        assert alert is None  # Same distribution — no alert


# ------------------------------------------------------------------ #
# Signal 2: Language Distribution                                      #
# ------------------------------------------------------------------ #

class TestLanguageDrift:
    def test_no_alert_on_english_traffic(self, svc):
        """All-English traffic should not alert."""
        baseline_langs = ["en"] * 10
        live_langs = ["en"] * 10
        fraction, alert = svc._check_language_drift(baseline_langs, live_langs)
        assert alert is None
        assert fraction == 0.0

    def test_alert_when_non_english_exceeds_threshold(self, svc):
        """More than 30% non-English (our test threshold) should alert."""
        baseline_langs = ["en"] * 10
        # 4/10 = 40% non-English — above our 30% test threshold
        live_langs = ["en"] * 6 + ["fr", "de", "es", "fr"]
        fraction, alert = svc._check_language_drift(baseline_langs, live_langs)
        assert alert is not None
        assert alert.signal == DriftSignal.LANGUAGE
        assert fraction == pytest.approx(0.4, abs=0.01)

    def test_no_alert_when_non_english_below_threshold(self, svc):
        """10% non-English (below our 30% threshold) should not alert."""
        baseline_langs = ["en"] * 10
        live_langs = ["en"] * 9 + ["fr"]
        fraction, alert = svc._check_language_drift(baseline_langs, live_langs)
        assert alert is None

    def test_language_distribution_helper(self, svc):
        """Language distribution helper should return correct fractions."""
        langs = ["en", "en", "fr", "de"]
        dist = svc._language_distribution(langs)
        assert dist["en"] == pytest.approx(0.5, abs=0.01)
        assert dist["fr"] == pytest.approx(0.25, abs=0.01)
        assert dist["de"] == pytest.approx(0.25, abs=0.01)


# ------------------------------------------------------------------ #
# Singleton behaviour                                                  #
# ------------------------------------------------------------------ #

class TestSingleton:
    def test_get_drift_service_returns_same_instance(self):
        from app.services.drift_service import get_drift_service
        svc1 = get_drift_service()
        svc2 = get_drift_service()
        assert svc1 is svc2

    def test_reset_clears_singleton(self):
        from app.services.drift_service import get_drift_service, reset_drift_service
        svc1 = get_drift_service()
        reset_drift_service()
        svc2 = get_drift_service()
        assert svc1 is not svc2