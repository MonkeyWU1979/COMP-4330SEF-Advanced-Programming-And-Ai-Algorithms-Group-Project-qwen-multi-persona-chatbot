"""
Evaluation engine for the prompt strategy comparison experiment.

The evaluator runs identical multi-turn scenarios under both the basic and
structured persona prompts, records response latency and token counts, and
persists both a flat CSV file (suitable for statistical analysis) and a
nested JSON file (which preserves the complete conversation history).

Qualitative 1-5 ratings are collected separately through the Gradio rating
panel or manual annotation; empty score columns are included in the CSV so
that ratings can be joined back to each turn later.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from .chat_engine import ChatEngine
from .config import (
    DEFAULT_EVALUATION_CONFIG,
    EVALUATION_DIMENSIONS,
    PROMPT_METHOD_BASIC,
    PROMPT_METHOD_STRUCTURED,
    EvaluationConfig,
)
from .model_manager import QwenModelManager
from .persona_manager import Persona

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class TurnRecord:
    """Flat representation of one evaluated conversation turn."""

    persona_key: str
    persona_name: str
    prompt_method: str
    repetition: int
    turn_index: int
    user_message: str
    assistant_response: str
    response_time_seconds: float
    prompt_token_count: int
    generated_token_count: int
    # Populated later through manual annotation
    persona_consistency: Optional[int] = None
    instruction_following: Optional[int] = None
    response_quality: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "persona_key": self.persona_key,
            "persona_name": self.persona_name,
            "prompt_method": self.prompt_method,
            "repetition": self.repetition,
            "turn_index": self.turn_index,
            "user_message": self.user_message,
            "assistant_response": self.assistant_response,
            "response_time_seconds": self.response_time_seconds,
            "prompt_token_count": self.prompt_token_count,
            "generated_token_count": self.generated_token_count,
            "persona_consistency": self.persona_consistency,
            "instruction_following": self.instruction_following,
            "response_quality": self.response_quality,
        }


@dataclass
class ExperimentSummary:
    """Aggregated latency and token statistics for one experiment run."""

    total_turns: int
    total_runtime_seconds: float
    records_by_condition: Dict[str, List[TurnRecord]] = field(default_factory=dict)

    def build_summary_dataframe(self) -> pd.DataFrame:
        """
        Return a DataFrame summarising mean latency and token counts per
        persona and prompt method.
        """
        rows: List[Dict[str, Any]] = []
        for condition, records in self.records_by_condition.items():
            persona_key, method = condition.split("|")
            latency_values = [r.response_time_seconds for r in records]
            generated_values = [r.generated_token_count for r in records]
            prompt_values = [r.prompt_token_count for r in records]

            rows.append(
                {
                    "persona_key": persona_key,
                    "prompt_method": method,
                    "num_turns": len(records),
                    "mean_response_time_s": round(
                        sum(latency_values) / len(latency_values), 4
                    ),
                    "min_response_time_s": round(min(latency_values), 4),
                    "max_response_time_s": round(max(latency_values), 4),
                    "mean_generated_tokens": round(
                        sum(generated_values) / len(generated_values), 2
                    ),
                    "mean_prompt_tokens": round(
                        sum(prompt_values) / len(prompt_values), 2
                    ),
                }
            )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
class ExperimentRunner:
    """Runs controlled comparisons between the two prompt strategies."""

    def __init__(
        self,
        model_manager: QwenModelManager,
        evaluation_config: Optional[EvaluationConfig] = None,
    ) -> None:
        self.model_manager = model_manager
        self.evaluation_config = evaluation_config or DEFAULT_EVALUATION_CONFIG
        self.records: List[TurnRecord] = []
        self.raw_conversations: List[Dict[str, Any]] = []

    def run_scenario(
        self,
        persona_key: str,
        persona: Persona,
        scenario_messages: List[str],
        prompt_methods: Optional[List[str]] = None,
        repetitions: Optional[int] = None,
    ) -> List[TurnRecord]:
        """
        Execute one persona's scenario across every requested prompt method.

        A fresh :class:`ChatEngine` is created for each condition so that
        conversation history from one strategy never leaks into another.
        """
        methods = prompt_methods or [PROMPT_METHOD_BASIC, PROMPT_METHOD_STRUCTURED]
        num_runs = repetitions or self.evaluation_config.num_repetitions
        produced_records: List[TurnRecord] = []

        for method in methods:
            for run_index in range(num_runs):
                logger.info(
                    "Running persona='%s' method='%s' repetition=%d",
                    persona_key,
                    method,
                    run_index,
                )
                engine = ChatEngine(
                    model_manager=self.model_manager,
                    persona=persona,
                    prompt_method=method,
                )

                conversation_payload: Dict[str, Any] = {
                    "persona_key": persona_key,
                    "persona_name": persona.name,
                    "prompt_method": method,
                    "repetition": run_index,
                    "turns": [],
                }

                for turn_index, user_text in enumerate(scenario_messages):
                    result = engine.send_message(user_text)
                    record = TurnRecord(
                        persona_key=persona_key,
                        persona_name=persona.name,
                        prompt_method=method,
                        repetition=run_index,
                        turn_index=turn_index,
                        user_message=user_text,
                        assistant_response=result.assistant_message,
                        response_time_seconds=result.response_time_seconds,
                        prompt_token_count=result.prompt_token_count,
                        generated_token_count=result.generated_token_count,
                    )
                    self.records.append(record)
                    produced_records.append(record)

                    conversation_payload["turns"].append(
                        {
                            "turn_index": turn_index,
                            "user_message": user_text,
                            "assistant_response": result.assistant_message,
                            "response_time_seconds": result.response_time_seconds,
                            "prompt_token_count": result.prompt_token_count,
                            "generated_token_count": result.generated_token_count,
                        }
                    )

                if self.evaluation_config.save_conversations:
                    self.raw_conversations.append(conversation_payload)

                # Explicitly drop the engine before the next condition
                del engine

        return produced_records

    def run_full_experiment(
        self,
        personas: Dict[str, Persona],
        scenarios: Dict[str, List[str]],
        prompt_methods: Optional[List[str]] = None,
        repetitions: Optional[int] = None,
    ) -> ExperimentSummary:
        """
        Run every available persona/scenario pair and persist all outputs.

        Returns an :class:`ExperimentSummary` containing the aggregated
        statistics.
        """
        start_time = datetime.now()
        common_keys = [key for key in personas.keys() if key in scenarios]
        missing = set(personas.keys()) ^ set(scenarios.keys())
        if missing:
            logger.warning(
                "Skipping personas without a matching scenario or vice versa: %s",
                missing,
            )

        # Ensure the model is loaded before the first timed run
        if not self.model_manager.is_loaded:
            self.model_manager.load_model()

        for persona_key in common_keys:
            self.run_scenario(
                persona_key=persona_key,
                persona=personas[persona_key],
                scenario_messages=scenarios[persona_key],
                prompt_methods=prompt_methods,
                repetitions=repetitions,
            )

        summary = self._build_summary()
        self.save_results(start_time=start_time)
        return summary

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_results(self, start_time: Optional[datetime] = None) -> Dict[str, str]:
        """
        Write the flat CSV, nested JSON and summary statistics files.

        Returns a mapping of output label to file path.
        """
        output_dir = self.evaluation_config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / self.evaluation_config.csv_filename
        json_path = output_dir / self.evaluation_config.json_filename
        summary_path = output_dir / "evaluation_summary.csv"

        records_df = pd.DataFrame([record.to_dict() for record in self.records])
        records_df.to_csv(csv_path, index=False, encoding="utf-8")

        payload = {
            "experiment_start": (start_time or datetime.now()).isoformat(timespec="seconds"),
            "experiment_end": datetime.now().isoformat(timespec="seconds"),
            "model_name": self.model_manager.model_config.model_name_or_path,
            "num_turn_records": len(self.records),
            "conversations": self.raw_conversations,
        }
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

        summary = self._build_summary()
        summary_df = summary.build_summary_dataframe()
        summary_df.to_csv(summary_path, index=False, encoding="utf-8")

        logger.info("Saved turn-level CSV: %s", csv_path)
        logger.info("Saved raw conversation JSON: %s", json_path)
        logger.info("Saved summary CSV: %s", summary_path)

        return {
            "csv": str(csv_path),
            "json": str(json_path),
            "summary": str(summary_path),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_summary(self) -> ExperimentSummary:
        grouped: Dict[str, List[TurnRecord]] = {}
        for record in self.records:
            key = f"{record.persona_key}|{record.prompt_method}"
            grouped.setdefault(key, []).append(record)

        total_runtime = sum(r.response_time_seconds for r in self.records)
        return ExperimentSummary(
            total_turns=len(self.records),
            total_runtime_seconds=total_runtime,
            records_by_condition=grouped,
        )

    def get_records_dataframe(self) -> pd.DataFrame:
        """Return all collected turn records as a pandas DataFrame."""
        return pd.DataFrame([record.to_dict() for record in self.records])
