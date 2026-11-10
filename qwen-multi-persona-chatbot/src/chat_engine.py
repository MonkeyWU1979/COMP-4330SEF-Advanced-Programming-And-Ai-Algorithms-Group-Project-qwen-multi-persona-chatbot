"""
Chat engine responsible for multi-turn dialogue management.

The engine combines the persona prompt builder, the Qwen model manager and a
sliding conversation history. It exposes a simple message-based API that is
used both by the Gradio web application and by the batch evaluation scripts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import (
    PROMPT_METHOD_BASIC,
    VALID_PROMPT_METHODS,
    GenerationConfig,
)
from .model_manager import ModelResponse, QwenModelManager
from .persona_manager import Persona, PersonaPromptBuilder

logger = logging.getLogger(__name__)


@dataclass
class ChatTurnResult:
    """Output returned after processing a single user turn."""

    user_message: str
    assistant_message: str
    response_time_seconds: float
    prompt_token_count: int
    generated_token_count: int
    persona_name: str
    prompt_method: str


class ChatEngine:
    """
    Stateful multi-turn chat engine.

    A single engine instance is bound to one model manager but can be
    reconfigured with different personas and prompt strategies at any time.
    """

    def __init__(
        self,
        model_manager: QwenModelManager,
        persona: Optional[Persona] = None,
        prompt_method: str = PROMPT_METHOD_BASIC,
        max_context_length: Optional[int] = None,
    ) -> None:
        if prompt_method not in VALID_PROMPT_METHODS:
            raise ValueError(
                f"Invalid prompt method '{prompt_method}'. "
                f"Expected one of: {', '.join(VALID_PROMPT_METHODS)}"
            )

        self.model_manager = model_manager
        self.prompt_method = prompt_method
        self.max_context_length = (
            max_context_length
            if max_context_length is not None
            else model_manager.model_config.max_context_length
        )

        # Conversation state
        self.persona: Optional[Persona] = None
        self.system_prompt: str = ""
        self.history: List[Dict[str, str]] = []

        if persona is not None:
            self.set_persona(persona, prompt_method)
        else:
            # Provide a neutral default persona so the engine is usable
            # immediately after construction.
            default_persona = Persona(
                name="Assistant",
                raw_description="You are a helpful, friendly conversational assistant.",
            )
            self.set_persona(default_persona, prompt_method)

    # ------------------------------------------------------------------
    # Persona and prompt configuration
    # ------------------------------------------------------------------
    def set_persona(
        self,
        persona: Persona,
        prompt_method: Optional[str] = None,
        clear_history: bool = False,
    ) -> None:
        """
        Assign a new persona and optionally switch the prompt strategy.

        Parameters
        ----------
        persona:
            The persona to adopt.
        prompt_method:
            Optional new prompt method. When omitted the current method is
            retained.
        clear_history:
            When True, all previous conversation turns are discarded. This is
            usually desirable when switching to an unrelated character.
        """
        if prompt_method is not None:
            if prompt_method not in VALID_PROMPT_METHODS:
                raise ValueError(
                    f"Invalid prompt method '{prompt_method}'. "
                    f"Expected one of: {', '.join(VALID_PROMPT_METHODS)}"
                )
            self.prompt_method = prompt_method

        self.persona = persona
        self.system_prompt = PersonaPromptBuilder.build_system_prompt(
            persona, self.prompt_method
        )

        if clear_history:
            self.history.clear()

        logger.info(
            "Persona set to '%s' using '%s' prompt (history cleared=%s)",
            persona.name,
            self.prompt_method,
            clear_history,
        )

    def update_prompt_method(
        self, prompt_method: str, clear_history: bool = False
    ) -> None:
        """
        Switch between basic and structured prompts while retaining the
        current persona definition.
        """
        if self.persona is None:
            raise RuntimeError("Cannot update prompt method before a persona is set.")
        self.set_persona(self.persona, prompt_method, clear_history)

    # ------------------------------------------------------------------
    # Conversation handling
    # ------------------------------------------------------------------
    def send_message(
        self,
        user_text: str,
        generation_config: Optional[GenerationConfig] = None,
    ) -> ChatTurnResult:
        """
        Append a user message, run model generation and append the reply.

        The engine automatically truncates older turns when the encoded
        conversation approaches the configured context length.
        """
        if self.model_manager is None:
            raise RuntimeError("Model manager is not initialised.")

        user_text = user_text.strip()
        if not user_text:
            raise ValueError("User message cannot be empty.")

        # Ensure the model is loaded before attempting generation
        if not self.model_manager.is_loaded:
            self.model_manager.load_model()

        self.history.append({"role": "user", "content": user_text})
        self._truncate_history_if_needed()

        messages = self._build_messages()
        model_response: ModelResponse = self.model_manager.generate(
            messages, generation_config=generation_config
        )

        self.history.append({"role": "assistant", "content": model_response.text})

        persona_name = self.persona.name if self.persona is not None else "Assistant"

        return ChatTurnResult(
            user_message=user_text,
            assistant_message=model_response.text,
            response_time_seconds=model_response.response_time_seconds,
            prompt_token_count=model_response.prompt_token_count,
            generated_token_count=model_response.generated_token_count,
            persona_name=persona_name,
            prompt_method=self.prompt_method,
        )

    def reset_conversation(self) -> None:
        """Clear all user/assistant turns while keeping the current persona."""
        self.history.clear()
        logger.info("Conversation history has been reset.")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_history(self) -> List[Dict[str, str]]:
        """Return a copy of the current user/assistant turn history."""
        return list(self.history)

    def get_gradio_chat_history(self) -> List[List[str]]:
        """
        Convert the internal message list into the legacy ``[user, assistant]``
        tuple format used by older Gradio releases.

        Gradio 6.x Chatbot uses the native message-dictionary format returned
        by :meth:`get_history`; this helper is retained for backward
        compatibility with older interfaces.
        """
        pairs: List[List[str]] = []
        current_user: Optional[str] = None

        for message in self.history:
            if message["role"] == "user":
                current_user = message["content"]
            elif message["role"] == "assistant" and current_user is not None:
                pairs.append([current_user, message["content"]])
                current_user = None

        return pairs

    def get_system_prompt(self) -> str:
        return self.system_prompt

    def _build_messages(self) -> List[Dict[str, str]]:
        """Combine the system prompt with the current conversation history."""
        return [
            {"role": "system", "content": self.system_prompt},
            *self.history,
        ]

    # ------------------------------------------------------------------
    # Context window management
    # ------------------------------------------------------------------
    def _truncate_history_if_needed(self) -> None:
        """
        Remove the oldest conversation pairs when the encoded context
        exceeds the allowed length.

        The system prompt is always retained. Truncation operates on whole
        user/assistant pairs to avoid leaving an orphaned turn.
        """
        tokenizer = self.model_manager.tokenizer
        if tokenizer is None:
            return

        # Reserve room for new tokens generated by the model
        allowed_prompt_tokens = max(
            256, self.max_context_length - self.model_manager.generation_config.max_new_tokens
        )

        while self.history:
            messages = self._build_messages()
            token_count = self._count_tokens(messages, tokenizer)
            if token_count <= allowed_prompt_tokens:
                break

            # Remove the oldest two messages (a user/assistant pair when
            # history is well-formed). If the first entry is somehow an
            # assistant message, remove just one entry.
            if len(self.history) >= 2 and self.history[0]["role"] == "user":
                del self.history[:2]
            else:
                del self.history[0]

        if self.history:
            logger.debug(
                "Context after truncation: %d messages", len(self.history)
            )

    @staticmethod
    def _count_tokens(messages: List[Dict[str, str]], tokenizer: Any) -> int:
        """Count tokens using the chat template without materialising tensors."""
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        return len(ids)
