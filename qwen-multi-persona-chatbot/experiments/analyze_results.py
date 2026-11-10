"""
Analysis utility for experiment outputs.

The script reads the turn-level CSV produced by ``run_evaluation.py`` and
computes descriptive statistics (mean, standard deviation, minimum and
maximum) for response time and token counts, grouped by prompt method and
persona. It also summarises manual ratings when the rating CSV exists.

Usage
-----

    python experiments/analyze_results.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


TURN_RESULTS_PATH = RESULTS_DIR / "evaluation_results.csv"
MANUAL_RATINGS_PATH = RESULTS_DIR / "manual_ratings.csv"
ANALYSIS_OUTPUT_PATH = RESULTS_DIR / "analysis_report.csv"


def load_turn_data() -> pd.DataFrame:
    """Load the turn-level evaluation CSV."""
    if not TURN_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Turn-level results not found at {TURN_RESULTS_PATH}. "
            "Run run_evaluation.py first."
        )
    return pd.read_csv(TURN_RESULTS_PATH)


def summarise_numeric_metrics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute descriptive statistics grouped by prompt method and by persona.

    Returns two DataFrames: an overall method-level summary and a
    persona-by-method summary.
    """
    numeric_columns = [
        "response_time_seconds",
        "prompt_token_count",
        "generated_token_count",
    ]

    method_summary = (
        df.groupby("prompt_method")[numeric_columns]
        .agg(["count", "mean", "std", "min", "max"])
        .round(4)
    )

    persona_summary = (
        df.groupby(["persona_key", "prompt_method"])[numeric_columns]
        .agg(["count", "mean", "std"])
        .round(4)
    )

    return method_summary, persona_summary


def summarise_qualitative_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise any populated qualitative rating columns in the turn data.

    Returns an empty DataFrame when no ratings have been annotated yet.
    """
    rating_columns = [
        "persona_consistency",
        "instruction_following",
        "response_quality",
    ]
    available = [col for col in rating_columns if col in df.columns]
    if not available or df[available].dropna(how="all").empty:
        return pd.DataFrame()

    return (
        df.groupby("prompt_method")[available]
        .agg(["count", "mean", "std"])
        .round(3)
    )


def summarise_manual_rating_file() -> pd.DataFrame:
    """Load and aggregate the standalone manual rating CSV when it exists."""
    if not MANUAL_RATINGS_PATH.exists():
        return pd.DataFrame()

    ratings = pd.read_csv(MANUAL_RATINGS_PATH)
    rating_columns = [
        "persona_consistency",
        "instruction_following",
        "response_quality",
    ]
    return (
        ratings.groupby("prompt_method")[rating_columns]
        .agg(["count", "mean", "std"])
        .round(3)
    )


def main() -> None:
    """Print all analyses and persist a flat analysis report."""
    df = load_turn_data()

    print("=" * 90)
    print("OVERALL METRICS BY PROMPT METHOD")
    print("=" * 90)
    method_summary, persona_summary = summarise_numeric_metrics(df)
    print(method_summary)

    print("\n" + "=" * 90)
    print("METRICS BY PERSONA AND PROMPT METHOD")
    print("=" * 90)
    print(persona_summary)

    qualitative_summary = summarise_qualitative_ratings(df)
    if not qualitative_summary.empty:
        print("\n" + "=" * 90)
        print("QUALITATIVE RATINGS (TURN-LEVEL)")
        print("=" * 90)
        print(qualitative_summary)

    manual_summary = summarise_manual_rating_file()
    if not manual_summary.empty:
        print("\n" + "=" * 90)
        print("MANUAL RATINGS FROM GRADIO SESSIONS")
        print("=" * 90)
        print(manual_summary)

    # Flatten the method-level summary for easy inclusion in the report
    flat_method_summary = method_summary.reset_index()
    flat_method_summary.to_csv(ANALYSIS_OUTPUT_PATH, index=False, encoding="utf-8")
    logger.info("Saved flattened analysis report to %s", ANALYSIS_OUTPUT_PATH)


if __name__ == "__main__":
    main()
