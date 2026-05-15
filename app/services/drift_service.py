"""
app/services/drift_service.py

Production drift monitoring stub — three independent signals.

What is "drift" and why does it matter?
  Your model was trained on a specific data distribution (SST-2 movie reviews,
  English, relatively short sentences). When the real-world inputs deviate from
  that distribution, model accuracy degrades — often silently, with no errors,
  no exceptions, just quietly wrong predictions.

  Drift monitoring answers: "Is what the model sees today similar to what
  it was trained on?" If the answer is no, you want to know before customers do.

Why a stub and not a full MLflow/Evidently integration?
  Two reasons:
    1. Free-tier compute. Evidently requires a separate server process.
    2. The signals themselves — a KS-test, a language ratio, and a rolling mean —
       are what demonstrate understanding. The library is just the vehicle.

  In a real system, you'd pipe these signals into:
    - Grafana (dashboarding) via Prometheus counters
    - PagerDuty (alerting) via webhook
    - Evidently AI (full drift reports) via a scheduled batch job

Architecture:
  DriftService is a singleton (module-level instance via get_drift_service()).
  It holds in-memory state: two deques (baseline window, live window).
  Every call to record() appends to the live window.
  Every DRIFT_WINDOW_SIZE calls, run_analysis() fires and compares.

  Why in-memory and not PostgreSQL for the windows?
    Speed. record() is called on the hot path (inside predict()).
    A deque.append() is O(1) and ~100ns.
    A DB write is ~10ms. We already write to DB in a BackgroundTask —
    the drift service should be faster than that, not slower.

    Tradeoff: if the process restarts, the window resets. Acceptable
    for a monitoring stub. In production, persist the baseline to Redis
    or a feature store so restarts don't lose calibration.
"""

import logging
import statistics
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ------------------------------------------------------------------ #
# Data Structures                                                      #
# ------------------------------------------------------------------ #


class DriftSignal(str, Enum):
    """Named drift signals. String enum so they serialise cleanly in JSON logs."""

    INPUT_LENGTH = "input_length_ks"
    LANGUAGE = "language_distribution"
    CONFIDENCE = "confidence_collapse"


@dataclass
class DriftAlert:
    """
    A single drift event. Logged as structured JSON when triggered.
    Also increments a Prometheus counter (if wired up in main.py).
    """

    signal: DriftSignal
    message: str
    current_value: float
    baseline_value: float
    threshold: float
    window_size: int
    severity: str = "WARNING"  # WARNING | CRITICAL

    def to_log_dict(self) -> dict:
        return {
            "drift_signal": self.signal.value,
            "detail": self.message,  # "message" is reserved by python-json-logger
            "current_value": round(self.current_value, 4),
            "baseline_value": round(self.baseline_value, 4),
            "threshold": self.threshold,
            "window_size": self.window_size,
            "severity": self.severity,
        }


@dataclass
class AnalysisResult:
    """
    The result of one full drift analysis pass.
    Contains all signal results so callers can log or act on them.
    """

    alerts: list[DriftAlert] = field(default_factory=list)
    window_size: int = 0
    baseline_size: int = 0
    is_drift_detected: bool = False

    # Individual signal values for observability (even when no alert)
    ks_pvalue: float | None = None
    language_non_english_fraction: float | None = None
    confidence_rolling_mean: float | None = None
    confidence_baseline_mean: float | None = None


# ------------------------------------------------------------------ #
# Drift Service                                                        #
# ------------------------------------------------------------------ #


class DriftService:
    """
    Stateful drift monitor. Holds two sliding windows:

      baseline_window: the first DRIFT_WINDOW_SIZE observations after
                       startup. This is your "known good" distribution.

      live_window:     the last DRIFT_WINDOW_SIZE observations. When this
                       fills, we run analysis and compare to baseline.

    Thread safety:
      record() and run_analysis() share mutable state (deques, flags).
      Uvicorn runs multiple async workers — without a lock, two concurrent
      requests could corrupt the deque mid-append.
      We use a threading.Lock (not asyncio.Lock) because record() is called
      from synchronous context (inside the async route handler but before
      any await). threading.Lock is safe here.
    """

    def __init__(self) -> None:
        self._window_size: int = settings.drift_window_size
        self._ks_threshold: float = settings.drift_ks_threshold
        self._confidence_drop_threshold: float = (
            settings.drift_confidence_drop_threshold
        )
        self._language_threshold: float = settings.drift_language_threshold

        # Baseline window: filled once, then frozen
        # maxlen ensures it never grows beyond window_size
        self._baseline_lengths: Deque[float] = deque(maxlen=self._window_size)
        self._baseline_confidences: Deque[float] = deque(maxlen=self._window_size)
        self._baseline_languages: Deque[str] = deque(maxlen=self._window_size)
        self._baseline_frozen: bool = False

        # Live window: continuously updated, reset after each analysis pass
        self._live_lengths: Deque[float] = deque(maxlen=self._window_size)
        self._live_confidences: Deque[float] = deque(maxlen=self._window_size)
        self._live_languages: Deque[str] = deque(maxlen=self._window_size)

        # Total observations since startup (for logging)
        self._total_observations: int = 0

        # Lock for thread safety
        self._lock = threading.Lock()

        logger.info(
            "DriftService initialised",
            extra={
                "window_size": self._window_size,
                "ks_threshold": self._ks_threshold,
                "confidence_drop_threshold": self._confidence_drop_threshold,
                "language_threshold": self._language_threshold,
            },
        )

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    def record(self, *, text: str, confidence: float) -> None:
        """
        Records one observation. Called on every non-cached prediction.

        Why not on cache hits?
          Cache hits don't represent new incoming data — they're replays
          of previous inputs. Including them would dilute the live window
          with stale data and make drift harder to detect.

        Language detection:
          langdetect is non-deterministic by default (uses random seed
          internally). We set a deterministic seed via DetectorFactory
          to ensure the same text always returns the same language.
          This matters for reproducibility in tests.
        """
        text_length = float(len(text))
        detected_lang = self._detect_language(text)

        with self._lock:
            self._total_observations += 1

            # -------------------------------------------------------- #
            # Phase 1: Fill baseline window (first N observations)      #
            # -------------------------------------------------------- #
            if not self._baseline_frozen:
                self._baseline_lengths.append(text_length)
                self._baseline_confidences.append(confidence)
                self._baseline_languages.append(detected_lang)

                if len(self._baseline_lengths) >= self._window_size:
                    self._baseline_frozen = True
                    baseline_conf_mean = statistics.mean(self._baseline_confidences)
                    logger.info(
                        "Drift baseline established",
                        extra={
                            "window_size": self._window_size,
                            "baseline_mean_length": round(
                                statistics.mean(self._baseline_lengths), 2
                            ),
                            "baseline_mean_confidence": round(baseline_conf_mean, 4),
                            "baseline_lang_dist": self._language_distribution(
                                self._baseline_languages
                            ),
                        },
                    )
                return  # Don't run analysis while building baseline

            # -------------------------------------------------------- #
            # Phase 2: Fill live window, run analysis when full         #
            # -------------------------------------------------------- #
            self._live_lengths.append(text_length)
            self._live_confidences.append(confidence)
            self._live_languages.append(detected_lang)

            if len(self._live_lengths) >= self._window_size:
                result = self._run_analysis()
                self._emit_alerts(result)
                # Reset live window for the next analysis cycle
                self._live_lengths.clear()
                self._live_confidences.clear()
                self._live_languages.clear()

    def get_status(self) -> dict:
        """
        Returns the current state of the drift monitor.
        Exposed via a /drift/status endpoint (wired up below).
        Useful for the walkthrough demo — shows the baseline is established
        and the live window is accumulating.
        """
        with self._lock:
            baseline_conf_mean = (
                statistics.mean(self._baseline_confidences)
                if self._baseline_confidences
                else None
            )
            live_conf_mean = (
                statistics.mean(self._live_confidences)
                if self._live_confidences
                else None
            )
            return {
                "baseline_established": self._baseline_frozen,
                "baseline_size": len(self._baseline_lengths),
                "live_window_size": len(self._live_lengths),
                "total_observations": self._total_observations,
                "baseline_mean_confidence": round(baseline_conf_mean, 4)
                if baseline_conf_mean
                else None,
                "live_mean_confidence": round(live_conf_mean, 4)
                if live_conf_mean
                else None,
                "window_capacity": self._window_size,
                "next_analysis_in": max(0, self._window_size - len(self._live_lengths)),
            }

    # ---------------------------------------------------------------- #
    # Signal 1: Input Length — KS Test                                  #
    # ---------------------------------------------------------------- #

    def _check_length_drift(
        self,
        baseline: list[float],
        live: list[float],
    ) -> tuple[float, DriftAlert | None]:
        """
        Two-sample Kolmogorov-Smirnov test on input length distributions.

        The KS test asks: "Could these two samples plausibly come from
        the same underlying distribution?" A low p-value (< threshold)
        means: "Almost certainly not — the distribution has shifted."

        Why KS and not a simple mean comparison?
          Mean comparison misses shape changes. If your baseline has
          lengths [50, 50, 50] and live has [10, 90, 50], the means
          are identical but the distributions are completely different.
          KS captures distribution shape, not just central tendency.

        Why input length as the signal?
          It's a proxy for content complexity and source changes:
          - Bots tend to send very short or very long inputs
          - A traffic source change often shifts length distribution
          - Prompt injection attempts are usually much longer than normal

        Returns (p_value, alert_or_None)
        """
        try:
            from scipy.stats import ks_2samp

            statistic, pvalue = ks_2samp(baseline, live)

            baseline_mean = statistics.mean(baseline)
            live_mean = statistics.mean(live)

            if pvalue < self._ks_threshold:
                alert = DriftAlert(
                    signal=DriftSignal.INPUT_LENGTH,
                    message=(
                        f"Input length distribution has shifted significantly. "
                        f"KS p-value={pvalue:.4f} (threshold={self._ks_threshold}). "
                        f"Baseline mean length={baseline_mean:.1f} chars, "
                        f"live mean={live_mean:.1f} chars."
                    ),
                    current_value=live_mean,
                    baseline_value=baseline_mean,
                    threshold=self._ks_threshold,
                    window_size=len(live),
                    severity="WARNING",
                )
                return pvalue, alert

            return pvalue, None

        except ImportError:
            logger.warning("scipy not installed — length drift check skipped")
            return 1.0, None
        except Exception as exc:
            logger.error("Length drift check failed", extra={"error": str(exc)})
            return 1.0, None

    # ---------------------------------------------------------------- #
    # Signal 2: Language Distribution                                   #
    # ---------------------------------------------------------------- #

    def _check_language_drift(
        self,
        baseline_langs: list[str],
        live_langs: list[str],
    ) -> tuple[float, DriftAlert | None]:
        """
        Checks whether the fraction of non-English requests has exceeded
        the configured threshold.

        Why this matters for this specific model:
          distilbert-base-uncased-finetuned-sst-2-english was fine-tuned
          exclusively on English text. Sending it French, Arabic, or Chinese
          inputs produces predictions that are technically valid (the model
          won't error) but meaningless — it's classifying subword tokens
          that have no semantic relationship to the original text.

          This is a silent failure mode. The model returns POSITIVE or NEGATIVE
          with high confidence on garbage input. Without this check, you'd never
          know your French-speaking users are getting random sentiment labels.

        Returns (non_english_fraction, alert_or_None)
        """
        if not live_langs:
            return 0.0, None

        baseline_non_english = sum(1 for lang in baseline_langs if lang != "en") / max(
            len(baseline_langs), 1
        )
        live_non_english = sum(1 for lang in live_langs if lang != "en") / len(
            live_langs
        )

        if live_non_english > self._language_threshold:
            lang_dist = self._language_distribution(live_langs)
            alert = DriftAlert(
                signal=DriftSignal.LANGUAGE,
                message=(
                    f"Non-English traffic fraction={live_non_english:.1%} "
                    f"exceeds threshold={self._language_threshold:.1%}. "
                    f"Model is English-only — predictions on non-English inputs are unreliable. "
                    f"Language breakdown: {lang_dist}"
                ),
                current_value=live_non_english,
                baseline_value=baseline_non_english,
                threshold=self._language_threshold,
                window_size=len(live_langs),
                severity="WARNING",
            )
            return live_non_english, alert

        return live_non_english, None

    # ---------------------------------------------------------------- #
    # Signal 3: Confidence Collapse                                     #
    # ---------------------------------------------------------------- #

    def _check_confidence_drift(
        self,
        baseline_confs: list[float],
        live_confs: list[float],
    ) -> tuple[float, float, DriftAlert | None]:
        """
        Checks whether mean model confidence has collapsed relative to baseline.

        Why confidence collapse predicts accuracy degradation:
          Softmax-based classifiers signal uncertainty through confidence.
          A well-calibrated model that has seen similar inputs before
          produces high-confidence predictions (0.95+).

          When the model encounters out-of-distribution inputs, confidence
          spreads toward 0.5 — it's uncertain. A collapsing mean confidence
          is therefore a leading indicator: it fires BEFORE accuracy drops
          enough for customers to notice.

          This is why it's more valuable than accuracy monitoring:
          - Accuracy monitoring requires ground truth labels (you don't have them in real-time)
          - Confidence monitoring requires nothing but the model's own output

          Analogy: a doctor's confidence in a diagnosis drops before they make
          an actual misdiagnosis. Watch the confidence, not just the diagnosis.

        Returns (baseline_mean, live_mean, alert_or_None)
        """
        if not baseline_confs or not live_confs:
            return 0.0, 0.0, None

        baseline_mean = statistics.mean(baseline_confs)
        live_mean = statistics.mean(live_confs)

        # Relative drop: how much has confidence fallen as a fraction of baseline?
        # e.g. baseline=0.95, live=0.80 → drop = (0.95-0.80)/0.95 = 15.8%
        relative_drop = (baseline_mean - live_mean) / max(baseline_mean, 1e-9)

        if relative_drop > self._confidence_drop_threshold:
            alert = DriftAlert(
                signal=DriftSignal.CONFIDENCE,
                message=(
                    f"Mean confidence collapsed by {relative_drop:.1%} "
                    f"(threshold={self._confidence_drop_threshold:.1%}). "
                    f"Baseline mean={baseline_mean:.4f}, "
                    f"current mean={live_mean:.4f}. "
                    f"Model may be encountering out-of-distribution inputs. "
                    f"Investigate input distribution and consider retraining."
                ),
                current_value=live_mean,
                baseline_value=baseline_mean,
                threshold=self._confidence_drop_threshold,
                window_size=len(live_confs),
                severity="CRITICAL" if relative_drop > 0.25 else "WARNING",
            )
            return baseline_mean, live_mean, alert

        return baseline_mean, live_mean, None

    # ---------------------------------------------------------------- #
    # Analysis Orchestrator                                             #
    # ---------------------------------------------------------------- #

    def _run_analysis(self) -> AnalysisResult:
        """
        Runs all three signals against the current live window.
        Called when the live window fills to DRIFT_WINDOW_SIZE.

        Note: called inside self._lock — do not acquire lock again here.
        All three signals are independent; we collect ALL alerts, not
        just the first one triggered.
        """
        baseline_lengths = list(self._baseline_lengths)
        baseline_confs = list(self._baseline_confidences)
        baseline_langs = list(self._baseline_languages)

        live_lengths = list(self._live_lengths)
        live_confs = list(self._live_confidences)
        live_langs = list(self._live_languages)

        result = AnalysisResult(
            window_size=len(live_lengths),
            baseline_size=len(baseline_lengths),
        )

        # Signal 1: length KS test
        pvalue, length_alert = self._check_length_drift(baseline_lengths, live_lengths)
        result.ks_pvalue = pvalue
        if length_alert:
            result.alerts.append(length_alert)

        # Signal 2: language distribution
        non_en_fraction, lang_alert = self._check_language_drift(
            baseline_langs, live_langs
        )
        result.language_non_english_fraction = non_en_fraction
        if lang_alert:
            result.alerts.append(lang_alert)

        # Signal 3: confidence collapse
        baseline_mean, live_mean, conf_alert = self._check_confidence_drift(
            baseline_confs, live_confs
        )
        result.confidence_baseline_mean = baseline_mean
        result.confidence_rolling_mean = live_mean
        if conf_alert:
            result.alerts.append(conf_alert)

        result.is_drift_detected = len(result.alerts) > 0
        return result

    def _emit_alerts(self, result: AnalysisResult) -> None:
        """
        Emits structured log lines for drift analysis results.
        Every analysis pass is logged — even clean passes — so you
        have a continuous audit trail of model health.
        """
        # Always log the analysis summary (at DEBUG level if clean)
        summary = {
            "event": "drift_analysis_complete",
            "total_observations": self._total_observations,
            "window_size": result.window_size,
            "drift_detected": result.is_drift_detected,
            "alert_count": len(result.alerts),
            "ks_pvalue": round(result.ks_pvalue, 4)
            if result.ks_pvalue is not None
            else None,
            "non_english_fraction": round(result.language_non_english_fraction, 4)
            if result.language_non_english_fraction is not None
            else None,
            "confidence_baseline": round(result.confidence_baseline_mean, 4)
            if result.confidence_baseline_mean is not None
            else None,
            "confidence_live": round(result.confidence_rolling_mean, 4)
            if result.confidence_rolling_mean is not None
            else None,
        }

        if result.is_drift_detected:
            # DRIFT_ALERT in the message makes it greppable:
            # grep 'DRIFT_ALERT' /var/log/app.log | jq .
            logger.warning("DRIFT_ALERT", extra=summary)
            for alert in result.alerts:
                logger.warning(
                    f"DRIFT_ALERT:{alert.signal.value}",
                    extra=alert.to_log_dict(),
                )
        else:
            logger.debug("drift_analysis_clean", extra=summary)

    # ---------------------------------------------------------------- #
    # Helpers                                                            #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _detect_language(text: str) -> str:
        """
        Detects language of text. Returns ISO 639-1 code (e.g. 'en', 'fr').
        Returns 'unknown' on failure.

        Uses DetectorFactory seed for determinism — critical for tests.
        langdetect is non-deterministic by default (random seed internally).
        """
        try:
            from langdetect import detect, DetectorFactory

            DetectorFactory.seed = 42  # Make deterministic
            return detect(text)
        except Exception:
            return "unknown"

    @staticmethod
    def _language_distribution(langs: Deque[str] | list[str]) -> dict[str, float]:
        """Returns language → fraction mapping for logging."""
        if not langs:
            return {}
        total = len(langs)
        counts: dict[str, int] = {}
        for lang in langs:
            counts[lang] = counts.get(lang, 0) + 1
        return {lang: round(count / total, 3) for lang, count in sorted(counts.items())}


# ------------------------------------------------------------------ #
# Singleton Factory                                                    #
# ------------------------------------------------------------------ #

_drift_service_instance: DriftService | None = None
_instance_lock = threading.Lock()


def get_drift_service() -> DriftService:
    """
    Returns the module-level DriftService singleton.

    Why a singleton?
      DriftService holds the baseline window in memory. If you create
      a new instance per request, the baseline never fills — you'd need
      100 simultaneous requests to the same object to trigger analysis.
      The singleton means all requests share the same accumulating window.

    Why not store on app.state (like ModelService)?
      DriftService has no async dependency — it doesn't need the event loop.
      A module-level singleton is simpler and avoids the Request dependency
      injection chain for what is essentially a background statistical process.

    Thread safety:
      Double-checked locking pattern. The inner check prevents a race
      condition where two threads both see _instance is None and both
      try to create one.
    """
    global _drift_service_instance
    if _drift_service_instance is None:
        with _instance_lock:
            if _drift_service_instance is None:
                _drift_service_instance = DriftService()
    return _drift_service_instance


def reset_drift_service() -> None:
    """
    Resets the singleton. Used in tests only — ensures each test
    starts with a clean drift state.
    """
    global _drift_service_instance
    with _instance_lock:
        _drift_service_instance = None
