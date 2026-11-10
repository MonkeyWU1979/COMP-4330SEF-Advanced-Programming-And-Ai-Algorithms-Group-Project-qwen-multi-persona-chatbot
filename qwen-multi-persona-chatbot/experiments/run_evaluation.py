"""
Command-line entry point for the comparative evaluation experiment.

Examples
--------
Run every persona under both prompt strategies:

    python experiments/run_evaluation.py

Run only two personas with three repeated runs:

    python experiments/run_evaluation.py --personas professor detective --repetitions 3

Disable 4-bit quantization (requires more VRAM):

    python experiments/run_evaluation.py --no-quantize
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    DEFAULT_GENERATION_CONFIG,
    ModelConfig,
    PROMPT_METHOD_BASIC,
    PROMPT_METHOD_STRUCTURED,
)
from src.evaluator import ExperimentRunner, ExperimentSummary  # noqa: E402
from src.model_manager import QwenModelManager  # noqa: E402

from experiments.persona_profiles import ALL_PERSONAS  # noqa: E402
from experiments.test_scenarios import SCENARIOS  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the experiment runner."""
    parser = argparse.ArgumentParser(
        description="Run the Qwen2.5-3B multi-persona prompt comparison experiment."
    )
    parser.add_argument(
        "--personas",
        nargs="+",
        choices=list(ALL_PERSONAS.keys()),
        default=list(ALL_PERSONAS.keys()),
        help="Subset of personas to evaluate. Defaults to all predefined personas.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[PROMPT_METHOD_BASIC, PROMPT_METHOD_STRUCTURED],
        default=[PROMPT_METHOD_BASIC, PROMPT_METHOD_STRUCTURED],
        help="Prompt strategies to compare.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of complete repeated runs per persona and method.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override the default Hugging Face model identifier or local model path.",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable 4-bit quantization. Requires substantially more VRAM.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the experiment according to the parsed CLI arguments."""
    args = parse_args()

    model_config = ModelConfig(load_in_4bit=not args.no_quantize)
    if args.model_name:
        model_config.model_name_or_path = args.model_name

    logger.info("Initialising model manager: %s", model_config.model_name_or_path)
    model_manager = QwenModelManager(
        model_config=model_config,
        generation_config=DEFAULT_GENERATION_CONFIG,
    )

    selected_personas = {
        key: ALL_PERSONAS[key] for key in args.personas if key in ALL_PERSONAS
    }
    selected_scenarios = {
        key: SCENARIOS[key] for key in args.personas if key in SCENARIOS
    }

    runner = ExperimentRunner(model_manager=model_manager)

    try:
        summary: ExperimentSummary = runner.run_full_experiment(
            personas=selected_personas,
            scenarios=selected_scenarios,
            prompt_methods=args.methods,
            repetitions=args.repetitions,
        )
    finally:
        # Keep the model loaded during the run; unload at the end to free
        # resources for subsequent applications.
        if model_manager.is_loaded:
            model_manager.unload_model()

    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    summary_df = summary.build_summary_dataframe()
    print(summary_df.to_string(index=False))
    print(
        f"\nTotal evaluated turns: {summary.total_turns} | "
        f"Total generation time: {summary.total_runtime_seconds:.2f} s"
    )


if __name__ == "__main__":
    main()
