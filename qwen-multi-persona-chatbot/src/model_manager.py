"""
Model manager for Qwen2.5-3B-Instruct.

Responsibilities
----------------
* Load the tokenizer and causal language model through Hugging Face
  Transformers.
* Configure 4-bit normalisation so the 3B model fits comfortably on an
  8 GB RTX 4060 laptop GPU.
* Apply the official Qwen chat template to multi-turn message lists.
* Generate responses and record wall-clock latency and token counts.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .config import GenerationConfig, ModelConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output data structure
# ---------------------------------------------------------------------------
@dataclass
class ModelResponse:
    """Structured output returned after every generation call."""

    text: str
    response_time_seconds: float
    prompt_token_count: int
    generated_token_count: int
    finish_reason: str = "stop"


# ---------------------------------------------------------------------------
# Model manager
# ---------------------------------------------------------------------------
class QwenModelManager:
    """
    Wrapper around the Qwen2.5-3B-Instruct model and tokenizer.

    The model is loaded lazily on the first call to :meth:`load_model` so
    that importing the module does not immediately allocate GPU memory.
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> None:
        self.model_config = model_config or ModelConfig()
        self.generation_config = generation_config or GenerationConfig()

        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._is_loaded: bool = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load_model(self) -> None:
        """
        Load the tokenizer and model into memory.

        Quantisation is configured through :class:`BitsAndBytesConfig` when
        ``load_in_4bit`` is enabled. The method is idempotent: repeated
        calls do not reload the model.
        """
        if self._is_loaded:
            logger.info("Model is already loaded; skipping reload.")
            return

        logger.info("Loading tokenizer from %s", self.model_config.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name_or_path,
            trust_remote_code=self.model_config.trust_remote_code,
            cache_dir=self.model_config.cache_dir,
        )

        # Some Qwen checkpoints do not define a pad token; use the EOS
        # token to avoid batch padding errors.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.model_config.trust_remote_code,
            "cache_dir": self.model_config.cache_dir,
        }

        if self.model_config.load_in_4bit:
            compute_dtype = _resolve_torch_dtype(self.model_config.bnb_4bit_compute_dtype)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.model_config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=self.model_config.bnb_4bit_use_double_quant,
            )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = self.model_config.device_map
            logger.info(
                "Loading model with 4-bit quantization (type=%s, compute=%s)",
                self.model_config.bnb_4bit_quant_type,
                self.model_config.bnb_4bit_compute_dtype,
            )
        else:
            model_kwargs["torch_dtype"] = _resolve_torch_dtype(
                self.model_config.torch_dtype
            )
            model_kwargs["device_map"] = self.model_config.device_map
            logger.info(
                "Loading model in full precision dtype=%s",
                self.model_config.torch_dtype,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name_or_path,
            **model_kwargs,
        )
        self.model.eval()

        # Sync the configured generation defaults with token IDs
        if self.generation_config.pad_token_id is None:
            self.generation_config.pad_token_id = self.tokenizer.pad_token_id
        if self.generation_config.eos_token_id is None:
            self.generation_config.eos_token_id = self.tokenizer.eos_token_id

        self._is_loaded = True
        logger.info("Model loaded successfully.")
        self._log_device_info()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        messages: List[Dict[str, str]],
        generation_config: Optional[GenerationConfig] = None,
    ) -> ModelResponse:
        """
        Generate a response for a chat-formatted message list.

        Parameters
        ----------
        messages:
            Chat messages in the OpenAI-style format
            ``[{"role": "system", "content": ...}, ...]``.
        generation_config:
            Optional per-call decoding parameters. When omitted, the
            instance-level default configuration is used.

        Returns
        -------
        ModelResponse
            The decoded response together with latency and token counts.
        """
        if not self._is_loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError("Model has not been loaded. Call load_model() first.")

        gen_config = generation_config or self.generation_config

        # Apply the model-specific chat template
        encoded_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(encoded_input, torch.Tensor):
            input_ids = encoded_input.to(self.model.device)
        else:
            # Defensive fallback for tokenizer implementations returning dicts
            input_ids = encoded_input["input_ids"].to(self.model.device)

        prompt_token_count = int(input_ids.shape[-1])

        # Synchronise GPU before starting the timer to capture only the
        # actual generation duration.
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

        if start_event is not None:
            start_event.record()

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=gen_config.max_new_tokens,
                temperature=gen_config.temperature,
                top_p=gen_config.top_p,
                top_k=gen_config.top_k,
                repetition_penalty=gen_config.repetition_penalty,
                do_sample=gen_config.do_sample,
                pad_token_id=gen_config.pad_token_id,
                eos_token_id=gen_config.eos_token_id,
            )

        if end_event is not None:
            end_event.record()
            torch.cuda.synchronize()
            response_time_seconds = start_event.elapsed_time(end_event) / 1000.0
        else:
            # CPU fallback (should not normally be used in this project)
            response_time_seconds = 0.0

        # Extract only the newly generated tokens
        generated_ids = output_ids[0, prompt_token_count:]
        response_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        generated_token_count = int(generated_ids.shape[-1])

        return ModelResponse(
            text=response_text,
            response_time_seconds=response_time_seconds,
            prompt_token_count=prompt_token_count,
            generated_token_count=generated_token_count,
        )

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------
    def unload_model(self) -> None:
        """Release model and tokenizer memory and clear the CUDA cache."""
        if not self._is_loaded:
            return

        del self.model
        self.model = None
        self.tokenizer = None
        self._is_loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logger.info("Model unloaded and GPU cache cleared.")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_device_summary(self) -> Dict[str, Any]:
        """Return basic device and memory information for logging."""
        if not torch.cuda.is_available():
            return {"cuda_available": False}

        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        allocated = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device_index) / (1024 ** 3)

        return {
            "cuda_available": True,
            "device_name": props.name,
            "total_vram_gb": round(props.total_memory / (1024 ** 3), 2),
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
        }

    def _log_device_info(self) -> None:
        summary = self.get_device_summary()
        if summary.get("cuda_available"):
            logger.info(
                "Device: %s | Total VRAM: %.2f GB | Allocated: %.2f GB",
                summary["device_name"],
                summary["total_vram_gb"],
                summary["allocated_gb"],
            )
        else:
            logger.warning("CUDA is not available; the model will run on CPU.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    """Map a string dtype name to the corresponding ``torch.dtype`` object."""
    mapping: Dict[str, torch.dtype] = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = dtype_name.lower()
    if key not in mapping:
        raise ValueError(
            f"Unsupported torch dtype '{dtype_name}'. "
            f"Valid options: {', '.join(mapping.keys())}"
        )
    return mapping[key]
