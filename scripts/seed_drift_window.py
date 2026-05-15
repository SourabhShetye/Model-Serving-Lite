"""
scripts/seed_drift_window.py

Pre-populates the drift monitor's baseline window by sending a batch
of real predictions to the /predict endpoint before going live.

Why this script exists:
  The DriftService requires DRIFT_WINDOW_SIZE non-cached predictions
  before it establishes a baseline. On a brand-new deployment, the
  first 100 requests come from real users — meaning drift alerts won't
  fire until 200 requests have been processed (100 to build baseline,
  100 to fill the first live window).

  This script sends a representative set of texts to the service
  immediately after deployment, establishing the baseline from known-good
  data rather than waiting for organic traffic.

When to run:
  1. After first deployment to a new environment
  2. After a process restart that cleared the in-memory baseline
  3. Before a live demo (ensures /drift/status shows baseline_established: true)

Usage:
  # Against local docker-compose stack
  python scripts/seed_drift_window.py --url http://localhost:8000

  # Against production
  python scripts/seed_drift_window.py --url https://sentiment-service.onrender.com

  # Custom window size (must match DRIFT_WINDOW_SIZE env var)
  python scripts/seed_drift_window.py --url http://localhost:8000 --count 100

  # Dry run — shows what would be sent without hitting the API
  python scripts/seed_drift_window.py --url http://localhost:8000 --dry-run
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from typing import Generator


# ------------------------------------------------------------------ #
# Representative seed texts                                            #
# ------------------------------------------------------------------ #
# Deliberately varied in length, sentiment, and complexity to produce
# a realistic baseline distribution rather than a degenerate one
# (e.g., all identical texts would give zero variance in the baseline,
# making any real traffic look like drift).

SEED_TEXTS = [
    # Short positive
    "Excellent product, highly recommend.",
    "Great quality and fast delivery.",
    "Love it, exactly what I needed.",
    "Works perfectly, very happy.",
    "Outstanding service, five stars.",

    # Long positive
    "I was initially sceptical but this exceeded every expectation I had. The build quality is superb and customer service responded within minutes when I had a question.",
    "After using this for three months I can confidently say it is the best purchase I have made this year. Would absolutely recommend to anyone looking for reliability.",
    "The attention to detail is remarkable. From packaging to performance, everything was thoughtfully considered. I will definitely be a returning customer.",

    # Short negative
    "Terrible quality, broke immediately.",
    "Complete waste of money.",
    "Very disappointed, avoid.",
    "Does not work as described.",
    "Poor customer service experience.",

    # Long negative
    "I have never been so disappointed with a product. It arrived damaged, stopped working after two days, and the returns process was an absolute nightmare that took three weeks.",
    "The product looks nothing like the photos. Build quality is cheap plastic, the instructions are incomprehensible, and it failed completely after first use.",
    "Shocking experience from start to finish. Late delivery, wrong item sent, and when I contacted support they were unhelpful and dismissive.",

    # Medium length, mixed
    "Good value for the price but the instructions could be clearer.",
    "Decent product overall, though the packaging left something to be desired.",
    "Not bad, but I expected better based on the reviews. Shipping was fast though.",
    "Does the job adequately. Nothing special but no major complaints either.",
    "Somewhere between satisfied and disappointed. Works but feels cheaply made.",
]


def check_service_ready(base_url: str, timeout: int = 30) -> bool:
    """
    Polls /ready until the service reports model_loaded: true or timeout.
    Returns True if ready, False if timed out.
    """
    ready_url = f"{base_url.rstrip('/')}/ready"
    deadline = time.time() + timeout
    print(f"Waiting for service to be ready at {ready_url}...")

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(ready_url, timeout=5) as resp:
                data = json.loads(resp.read())
                if data.get("model_loaded"):
                    print(f"  ✓ Service ready (uptime: {data.get('uptime_seconds', '?')}s)")
                    return True
                print(f"  … model_loaded={data.get('model_loaded')} — waiting")
        except urllib.error.HTTPError as e:
            if e.code == 503:
                print("  … 503 Not Ready — model still loading")
            else:
                print(f"  … HTTP {e.code} — retrying")
        except Exception as e:
            print(f"  … {e} — retrying")
        time.sleep(3)

    print(f"Timed out after {timeout}s waiting for service to be ready.")
    return False


def get_drift_status(base_url: str) -> dict:
    """Fetches current drift monitor status."""
    url = f"{base_url.rstrip('/')}/drift/status"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def send_prediction(base_url: str, text: str) -> dict:
    """Sends one prediction request. Returns the response dict."""
    url = f"{base_url.rstrip('/')}/predict/"
    payload = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def seed_texts_generator(seed_texts: list[str], count: int) -> Generator[str, None, None]:
    """
    Yields `count` texts by cycling through seed_texts.
    Ensures variety even when count > len(seed_texts).
    """
    for i in range(count):
        yield seed_texts[i % len(seed_texts)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-seed the drift monitor baseline window with representative predictions."
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the sentiment service (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=110,
        help=(
            "Number of predictions to send. Should exceed DRIFT_WINDOW_SIZE "
            "(default 100) to fully establish the baseline. Default: 110."
        ),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="Seconds to wait between requests (default: 0.05). Increase for rate-limited services.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be sent without making any API calls.",
    )
    parser.add_argument(
        "--skip-ready-check",
        action="store_true",
        help="Skip the /ready poll and send requests immediately.",
    )
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    print(f"\n{'='*60}")
    print(f"  Drift Baseline Seeder")
    print(f"  Target:  {base_url}")
    print(f"  Sending: {args.count} predictions")
    print(f"  Delay:   {args.delay}s between requests")
    print(f"  Dry run: {args.dry_run}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("DRY RUN — no requests will be sent. Texts that would be sent:\n")
        for i, text in enumerate(seed_texts_generator(SEED_TEXTS, args.count), 1):
            print(f"  {i:3d}. {text[:70]}{'...' if len(text) > 70 else ''}")
        print(f"\nTotal: {args.count} requests. Remove --dry-run to execute.")
        sys.exit(0)

    # ---------------------------------------------------------------- #
    # 1. Wait for service to be ready                                   #
    # ---------------------------------------------------------------- #
    if not args.skip_ready_check:
        if not check_service_ready(base_url, timeout=60):
            print("ERROR: Service did not become ready in time. Aborting.")
            sys.exit(1)

    # ---------------------------------------------------------------- #
    # 2. Check current drift status before seeding                     #
    # ---------------------------------------------------------------- #
    status_before = get_drift_status(base_url)
    print(f"Drift status BEFORE seeding:")
    print(f"  baseline_established : {status_before.get('baseline_established', '?')}")
    print(f"  baseline_size        : {status_before.get('baseline_size', '?')}")
    print(f"  total_observations   : {status_before.get('total_observations', '?')}")
    print()

    if status_before.get("baseline_established"):
        print("Baseline already established — seeding anyway to grow the total observations.\n")

    # ---------------------------------------------------------------- #
    # 3. Send predictions                                               #
    # ---------------------------------------------------------------- #
    success_count = 0
    cache_hits = 0
    errors = 0
    t_start = time.perf_counter()

    for i, text in enumerate(seed_texts_generator(SEED_TEXTS, args.count), 1):
        try:
            result = send_prediction(base_url, text)
            success_count += 1
            if result.get("cache_hit"):
                cache_hits += 1

            # Progress indicator every 10 requests
            if i % 10 == 0 or i == args.count:
                elapsed = time.perf_counter() - t_start
                rps = i / elapsed
                print(
                    f"  [{i:3d}/{args.count}]  "
                    f"label={result.get('label', '?'):8s}  "
                    f"confidence={result.get('confidence', 0):.4f}  "
                    f"cache_hit={str(result.get('cache_hit', '?')):5s}  "
                    f"({rps:.1f} req/s)"
                )

        except urllib.error.HTTPError as e:
            errors += 1
            print(f"  [{i:3d}/{args.count}]  HTTP ERROR {e.code}: {e.reason}")
        except Exception as e:
            errors += 1
            print(f"  [{i:3d}/{args.count}]  ERROR: {e}")

        if args.delay > 0:
            time.sleep(args.delay)

    # ---------------------------------------------------------------- #
    # 4. Summary                                                        #
    # ---------------------------------------------------------------- #
    elapsed_total = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"  Seeding complete")
    print(f"  Sent:       {success_count} / {args.count} successful")
    print(f"  Cache hits: {cache_hits} (expected — seed texts repeat)")
    print(f"  Errors:     {errors}")
    print(f"  Duration:   {elapsed_total:.1f}s  ({success_count/elapsed_total:.1f} req/s)")
    print(f"{'='*60}\n")

    # ---------------------------------------------------------------- #
    # 5. Check drift status after seeding                               #
    # ---------------------------------------------------------------- #
    time.sleep(1)  # Brief pause for any async writes to settle
    status_after = get_drift_status(base_url)
    print(f"Drift status AFTER seeding:")
    print(f"  baseline_established : {status_after.get('baseline_established', '?')}")
    print(f"  baseline_size        : {status_after.get('baseline_size', '?')}")
    print(f"  live_window_size     : {status_after.get('live_window_size', '?')}")
    print(f"  total_observations   : {status_after.get('total_observations', '?')}")
    print(f"  baseline_mean_conf   : {status_after.get('baseline_mean_confidence', '?')}")
    print()

    if status_after.get("baseline_established"):
        print("✓ Baseline established. Drift monitoring is active.")
        print("  Run: curl {}/drift/status | jq .".format(base_url))
    else:
        needed = status_after.get("window_capacity", 100) - status_after.get("baseline_size", 0)
        print(f"✗ Baseline NOT yet established. Need {needed} more non-cached predictions.")
        print(f"  Re-run with --count {needed + 10} to complete seeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
