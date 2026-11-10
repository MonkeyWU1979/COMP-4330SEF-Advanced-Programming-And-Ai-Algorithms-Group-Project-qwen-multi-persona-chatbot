"""
Persona configuration module.

This module transforms user-supplied character descriptions into system
prompts that can be consumed by Qwen2.5-3B-Instruct. Two prompt strategies
are implemented exactly as described in the project proposal:

1. Basic Prompt
   The raw character definition is forwarded to the model with only minimal
   wrapping instructions.

2. Structured Persona Prompt
   The character is decomposed into explicit sections covering personality,
   speaking style, background and behavioural traits, with strong
   role-maintenance instructions.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from .config import (
    PROMPT_METHOD_BASIC,
    PROMPT_METHOD_STRUCTURED,
    VALID_PROMPT_METHODS,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Persona:
    """
    Structured representation of a chatbot persona.

    A persona can be created either from individual structured fields or
    from a single free-text description. Missing fields are simply omitted
    when constructing the structured prompt.
    """

    name: str = "Assistant"
    raw_description: str = ""
    personality_traits: List[str] = field(default_factory=list)
    speaking_style: str = ""
    background: str = ""
    behavioral_traits: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_raw_description(
        cls, name: str, raw_description: str
    ) -> "Persona":
        """Create a persona from a single free-text character description."""
        return cls(name=name.strip() or "Assistant", raw_description=raw_description.strip())

    @classmethod
    def from_structured_fields(
        cls,
        name: str,
        personality_traits: str,
        speaking_style: str,
        background: str,
        behavioral_traits: str,
    ) -> "Persona":
        """
        Create a persona from comma-separated structured input fields.

        Empty strings are converted to empty lists rather than a list
        containing a single empty string.
        """
        def _split(value: str) -> List[str]:
            return [item.strip() for item in value.split(",") if item.strip()]

        return cls(
            name=name.strip() or "Assistant",
            personality_traits=_split(personality_traits),
            speaking_style=speaking_style.strip(),
            background=background.strip(),
            behavioral_traits=_split(behavioral_traits),
        )


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
class PersonaPromptBuilder:
    """Builds system prompts for the two personalisation strategies."""

    # ------------------------------------------------------------------
    # Basic prompt: minimal additional instructions
    # ------------------------------------------------------------------
    @staticmethod
    def build_basic_prompt(persona: Persona) -> str:
        """
        Construct a basic prompt that forwards the user's character
        definition with very little extra instruction.
        """
        if persona.raw_description:
            character_text = persona.raw_description
        else:
            # Fall back to concatenating structured fields when no raw text
            # is available, still keeping wrapping instructions minimal.
            character_text = _format_structured_character(persona)

        return (
            f"You are {persona.name}.\n"
            f"{character_text}\n"
            f"Respond to the user while staying in character as {persona.name}."
        )

    # ------------------------------------------------------------------
    # Structured persona prompt: explicit sections and strong constraints
    # ------------------------------------------------------------------
    @staticmethod
    def build_structured_prompt(persona: Persona) -> str:
        """
        Construct a structured prompt that organises character information
        into clearly labelled sections and adds explicit role-maintenance
        rules.
        """
        sections: List[str] = [
            "You are an AI role-playing assistant. You must fully adopt and "
            "maintain the persona defined below for the entire conversation.",
            f"# Character Name\n{persona.name}",
        ]

        # Preserve the user's raw description when available
        if persona.raw_description:
            sections.append(f"# Character Description\n{persona.raw_description}")

        if persona.personality_traits:
            traits = "\n".join(f"- {trait}" for trait in persona.personality_traits)
            sections.append(f"# Personality Traits\n{traits}")

        if persona.background:
            sections.append(f"# Background\n{persona.background}")

        if persona.speaking_style:
            sections.append(f"# Speaking Style\n{persona.speaking_style}")

        if persona.behavioral_traits:
            behaviors = "\n".join(f"- {item}" for item in persona.behavioral_traits)
            sections.append(f"# Behavioral Rules\n{behaviors}")

        sections.append(
            "# Role-Playing Requirements\n"
            "1. Always respond in the first person as this character; never "
            "refer to yourself as an AI or language model.\n"
            "2. Keep your personality, tone and speaking style consistent "
            "with the definition above in every turn.\n"
            "3. Maintain context across multi-turn conversations and do not "
            "break character unless the user explicitly asks you to stop.\n"
            "4. If a request conflicts with the persona, respond in a way "
            "that the character would naturally respond rather than "
            "abandoning the role."
        )

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Public dispatch method
    # ------------------------------------------------------------------
    @classmethod
    def build_system_prompt(cls, persona: Persona, method: str) -> str:
        """
        Return the appropriate system prompt for the requested strategy.

        Parameters
        ----------
        persona:
            The persona to encode.
        method:
            One of ``"basic"`` or ``"structured"``.

        Raises
        ------
        ValueError
            If the supplied method is not a recognised prompt strategy.
        """
        if method not in VALID_PROMPT_METHODS:
            raise ValueError(
                f"Unknown prompt method '{method}'. "
                f"Expected one of: {', '.join(VALID_PROMPT_METHODS)}"
            )

        if method == PROMPT_METHOD_BASIC:
            return cls.build_basic_prompt(persona)
        return cls.build_structured_prompt(persona)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _format_structured_character(persona: Persona) -> str:
    """Concatenate structured fields into plain text for the basic prompt."""
    parts: List[str] = []

    if persona.personality_traits:
        parts.append("Personality: " + ", ".join(persona.personality_traits))
    if persona.background:
        parts.append("Background: " + persona.background)
    if persona.speaking_style:
        parts.append("Speaking style: " + persona.speaking_style)
    if persona.behavioral_traits:
        parts.append("Behavior: " + ", ".join(persona.behavioral_traits))

    return "\n".join(parts) if parts else "You are a helpful conversational assistant."
