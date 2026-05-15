"""
app/routers/drift.py

Drift monitoring endpoints.

Two endpoints:
  GET  /drift/status   — Shows current window state (baseline established?
                          how full is the live window? current confidence mean?)
                          Used in the walkthrough demo to show the monitor is live.

  POST /drift/simulate — Injects synthetic out-of-distribution samples to
                          trigger a drift alert on demand.
                          Used in the walkthrough demo for "simulating drift".
                          Would NOT exist in a real production service.
"""

import logging
import random
import string

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.services.drift_service import get_drift_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/drift", tags=["observability"])


class DriftStatusResponse(BaseModel):
    baseline_established: bool
    baseline_size: int
    live_window_size: int
    window_capacity: int
    next_analysis_in: int
    total_observations: int
    baseline_mean_confidence: float | None
    live_mean_confidence: float | None
    message: str


class SimulateRequest(BaseModel):
    scenario: str = Field(
        default="confidence_collapse",
        description=(
            "Which drift scenario to simulate. "
            "Options: 'confidence_collapse', 'length_shift', 'language_shift'"
        ),
    )
    num_samples: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Number of synthetic samples to inject into the live window.",
    )


class SimulateResponse(BaseModel):
    scenario: str
    samples_injected: int
    message: str


@router.get(
    "/status",
    response_model=DriftStatusResponse,
    summary="Current state of the drift monitor",
)
async def drift_status() -> DriftStatusResponse:
    """
    Returns the current state of the drift monitoring windows.

    Useful for:
      - Verifying the baseline has been established (needs DRIFT_WINDOW_SIZE predictions)
      - Watching the live window fill in real-time
      - Checking current vs baseline mean confidence
    """
    drift_svc = get_drift_service()
    status_data = drift_svc.get_status()

    if not status_data["baseline_established"]:
        message = (
            f"Baseline not yet established. "
            f"Need {status_data['window_capacity'] - status_data['baseline_size']} "
            f"more non-cached predictions."
        )
    elif status_data["next_analysis_in"] > 0:
        message = (
            f"Baseline established. "
            f"Next analysis in {status_data['next_analysis_in']} predictions."
        )
    else:
        message = "Analysis running — live window just completed."

    return DriftStatusResponse(**status_data, message=message)


@router.post(
    "/simulate",
    response_model=SimulateResponse,
    status_code=status.HTTP_200_OK,
    summary="Inject synthetic drift samples for demo/testing",
)
async def simulate_drift(body: SimulateRequest) -> SimulateResponse:
    """
    Injects synthetic samples to trigger a drift alert.

    **For walkthrough demo use only.**

    Scenarios:
    - `confidence_collapse`: injects low-confidence predictions (0.51–0.59)
    - `length_shift`: injects very long synthetic texts (500–512 chars)
    - `language_shift`: injects short texts detected as non-English

    After injection, the next analysis cycle will fire and emit a
    DRIFT_ALERT log line. Watch for it in the server logs.
    """
    valid_scenarios = {"confidence_collapse", "length_shift", "language_shift"}
    if body.scenario not in valid_scenarios:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown scenario '{body.scenario}'. Choose from: {valid_scenarios}",
        )

    drift_svc = get_drift_service()

    if not drift_svc.get_status()["baseline_established"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Baseline not yet established. "
                "Send enough predictions to /predict first to fill the baseline window, "
                "then simulate drift."
            ),
        )

    injected = 0

    if body.scenario == "confidence_collapse":
        # Inject predictions with low confidence (near-random model output)
        for _ in range(body.num_samples):
            synthetic_text = "This is a test sentence with uncertain sentiment."
            low_confidence = random.uniform(0.51, 0.59)
            drift_svc.record(text=synthetic_text, confidence=low_confidence)
            injected += 1
        message = (
            f"Injected {injected} low-confidence samples (confidence ~0.51–0.59). "
            f"Watch server logs for DRIFT_ALERT:confidence_collapse when the window fills."
        )

    elif body.scenario == "length_shift":
        # Inject very long synthetic texts — character distribution shifts dramatically
        for _ in range(body.num_samples):
            synthetic_text = "".join(
                random.choices(string.ascii_lowercase + " ", k=random.randint(400, 512))
            )
            confidence = random.uniform(
                0.85, 0.99
            )  # High confidence — only length shifts
            drift_svc.record(text=synthetic_text, confidence=confidence)
            injected += 1
        message = (
            f"Injected {injected} long-text samples (length ~400–512 chars). "
            f"Watch server logs for DRIFT_ALERT:input_length_ks when the window fills."
        )

    elif body.scenario == "language_shift":
        # Inject texts in non-English languages
        # These are real phrases — langdetect will detect them correctly
        non_english_samples = [
            "C'est une très bonne journée aujourd'hui.",  # French
            "Das Wetter ist heute wirklich wunderschön.",  # German
            "Este producto es absolutamente increíble y útil.",  # Spanish
            "Il prodotto è di ottima qualità e mi è piaciuto.",  # Italian
            "Este filme foi uma experiência incrível para mim.",  # Portuguese
        ]
        for i in range(body.num_samples):
            synthetic_text = non_english_samples[i % len(non_english_samples)]
            confidence = random.uniform(0.80, 0.95)
            drift_svc.record(text=synthetic_text, confidence=confidence)
            injected += 1
        message = (
            f"Injected {injected} non-English samples (FR/DE/ES/IT/PT). "
            f"Watch server logs for DRIFT_ALERT:language_distribution when the window fills."
        )

    logger.info(
        "Drift simulation injected",
        extra={"scenario": body.scenario, "samples_injected": injected},
    )

    return SimulateResponse(
        scenario=body.scenario,
        samples_injected=injected,
        message=message,
    )
