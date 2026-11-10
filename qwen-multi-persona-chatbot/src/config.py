"""
Global configuration for the Qwen2.5 multi-persona chatbot project.

All model paths, default generation parameters and directory constants are
centralised here so that experiments can be reproduced without editing
individual modules.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment loading and Hugging Face authentication
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Load variables from a local .env file if python-dotenv is available.
# This is wrapped in a try block so the project still runs without the
# optional dependency (real environment variables are then used instead).
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # pragma: no cover - optional dependency
    pass


def authenticate_huggingface() -> Optional[str]:
    """
    Authenticate with the Hugging Face Hub using the HF_TOKEN environment
    variable.

    Returns the token when authentication succeeds, ``None`` when no token
    is configured. The token itself is never printed to logs.
    """
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        logger.info(
            "No HF_TOKEN found; using anonymous Hugging Face requests "
            "(lower rate limits)."
        )
        return None

    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
        logger.info("Authenticated with Hugging Face Hub using HF_TOKEN.")
        return token
    except Exception as exc:  # pragma: no cover - surfaced as a warning
        logger.warning("Hugging Face authentication failed: %s", exc)
        return None


# Authenticate once at import time so every entry point (app and experiments)
# benefits from the higher authenticated rate limits.
HF_TOKEN = authenticate_huggingface()


# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
DATA_DIR: Path = PROJECT_ROOT / "data"
RESULTS_DIR: Path = DATA_DIR / "evaluation_results"

# Ensure output directories exist at import time
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME: str = "Qwen/Qwen2.5-3B-Instruct"

# Prompt strategies compared in the study
PROMPT_METHOD_BASIC: str = "basic"
PROMPT_METHOD_STRUCTURED: str = "structured"
VALID_PROMPT_METHODS: tuple = (PROMPT_METHOD_BASIC, PROMPT_METHOD_STRUCTURED)


@dataclass
class ModelConfig:
    """Settings controlling model loading and memory optimisation."""

    model_name_or_path: str = DEFAULT_MODEL_NAME
    device_map: str = "auto"
    # 4-bit normalisation dramatically reduces VRAM usage on RTX 4060 (8 GB)
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    # Fallback dtype when 4-bit loading is disabled
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    # Maximum context length the chat engine will retain
    max_context_length: int = 4096
    # Local cache directory for downloaded model weights (optional)
    cache_dir: Optional[str] = field(default=None)


@dataclass
class GenerationConfig:
    """Default decoding parameters used during chat and evaluation."""

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 20
    repetition_penalty: float = 1.05
    do_sample: bool = True
    # No specific stop strings are required for Qwen chat templates
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------------
# Rating scale defined in the proposal: 1 (worst) to 5 (best)
MIN_RATING: int = 1
MAX_RATING: int = 5

EVALUATION_DIMENSIONS: tuple = (
    "persona_consistency",
    "instruction_following",
    "response_quality",
)


@dataclass
class EvaluationConfig:
    """Configuration for batch experiment runs."""

    output_dir: Path = RESULTS_DIR
    # Number of independent runs per scenario for stable latency measurement
    num_repetitions: int = 1
    # Whether to persist raw responses alongside numeric scores
    save_conversations: bool = True
    csv_filename: str = "evaluation_results.csv"
    json_filename: str = "evaluation_results.json"


# Singleton instances imported by other modules
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_GENERATION_CONFIG = GenerationConfig()
DEFAULT_EVALUATION_CONFIG = EvaluationConfig()
