"""
Gradio web application for the Qwen2.5-3B multi-persona chatbot.

Run the application with:

    python app.py

The browser interface supports persona customisation, switching between the
basic and structured prompt strategies, adjusting decoding parameters,
running multi-turn conversations, and manually rating responses on the
three qualitative dimensions defined in the proposal.
"""

from __future__ import annotations

import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr

# Ensure the project root is importable when launched directly
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chat_engine import ChatEngine, ChatTurnResult
from src.config import (
    DEFAULT_GENERATION_CONFIG,
    DEFAULT_MODEL_CONFIG,
    EVALUATION_DIMENSIONS,
    MAX_RATING,
    MIN_RATING,
    PROMPT_METHOD_BASIC,
    PROMPT_METHOD_STRUCTURED,
    RESULTS_DIR,
    GenerationConfig,
)
from src.model_manager import QwenModelManager
from src.persona_manager import Persona

# Predefined personas live inside the experiments package
from experiments.persona_profiles import ALL_PERSONAS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global engine initialisation (model is loaded lazily on first message)
# ---------------------------------------------------------------------------
model_manager = QwenModelManager(
    model_config=DEFAULT_MODEL_CONFIG,
    generation_config=DEFAULT_GENERATION_CONFIG,
)
chat_engine = ChatEngine(
    model_manager=model_manager,
    prompt_method=PROMPT_METHOD_BASIC,
)

MANUAL_RATING_FILE = RESULTS_DIR / "manual_ratings.csv"


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------
def load_predefined_persona(persona_key: str) -> Tuple:
    """
    Populate the persona form fields from a predefined profile.

    Returns values for every editable configuration component so the UI
    stays in sync with the selected profile.
    """
    if not persona_key or persona_key not in ALL_PERSONAS:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    persona = ALL_PERSONAS[persona_key]
    return (
        persona.name,
        persona.raw_description,
        ", ".join(persona.personality_traits),
        persona.speaking_style,
        persona.background,
        ", ".join(persona.behavioral_traits),
    )


def apply_persona(
    name: str,
    raw_description: str,
    personality_traits: str,
    speaking_style: str,
    background: str,
    behavioral_traits: str,
    prompt_method: str,
    clear_history_on_apply: bool,
) -> Tuple[str, str, List]:
    """
    Build a persona from the form inputs and update the chat engine.

    The structured fields are always used to construct the persona; the
    free-text description is also retained so the basic prompt can forward
    it verbatim when provided.
    """
    if not raw_description.strip() and not any(
        [personality_traits.strip(), speaking_style.strip(), background.strip(), behavioral_traits.strip()]
    ):
        return (
            "Please provide at least a free-text description or structured persona fields.",
            "",
            chat_engine.get_history(),
        )

    persona = Persona.from_structured_fields(
        name=name,
        personality_traits=personality_traits,
        speaking_style=speaking_style,
        background=background,
        behavioral_traits=behavioral_traits,
    )
    # Preserve the raw description in addition to structured fields
    persona.raw_description = raw_description.strip()

    chat_engine.set_persona(
        persona,
        prompt_method=prompt_method,
        clear_history=bool(clear_history_on_apply),
    )

    status = (
        f"Persona '{persona.name}' applied successfully using the "
        f"'{prompt_method}' prompt strategy."
    )
    if clear_history_on_apply:
        status += " Conversation history was cleared."

    return status, chat_engine.get_system_prompt(), chat_engine.get_history()


def respond(
    user_message: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    repetition_penalty: float,
) -> Tuple[List, str, str]:
    """
    Send a user message through the chat engine and return UI updates.

    The model is loaded lazily here so that launching the app does not
    block on weight downloads before the UI appears.
    """
    user_message = (user_message or "").strip()
    if not user_message:
        return (
            chat_engine.get_history(),
            "",
            "Please enter a message before sending.",
        )

    try:
        if not model_manager.is_loaded:
            logger.info("Loading model on first request ...")
            model_manager.load_model()

        generation_config = GenerationConfig(
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
            do_sample=True,
            pad_token_id=model_manager.generation_config.pad_token_id,
            eos_token_id=model_manager.generation_config.eos_token_id,
        )

        result: ChatTurnResult = chat_engine.send_message(
            user_message, generation_config=generation_config
        )
    except Exception as exc:  # pragma: no cover - surfaced in the UI
        logger.exception("Generation failed")
        return (
            chat_engine.get_history(),
            "",
            f"Error during generation: {exc}",
        )

    info_text = (
        f"Response time: {result.response_time_seconds:.3f} s | "
        f"Prompt tokens: {result.prompt_token_count} | "
        f"Generated tokens: {result.generated_token_count} | "
        f"Persona: {result.persona_name} | Method: {result.prompt_method}"
    )
    return chat_engine.get_history(), "", info_text


def reset_conversation() -> Tuple[List, str]:
    """Clear the chat history and reset the response info panel."""
    chat_engine.reset_conversation()
    return [], "Conversation has been reset. The current persona is retained."


def save_manual_rating(
    persona_consistency: int,
    instruction_following: int,
    response_quality: int,
    comment: str,
) -> str:
    """
    Persist a 1-5 manual rating for the most recent assistant response.

    Ratings are appended to a CSV file so that multiple experiment sessions
    can be aggregated later.
    """
    if not chat_engine.history:
        return "No conversation is available to rate yet."

    last_assistant_turn = None
    last_user_turn = None
    for message in reversed(chat_engine.history):
        if message["role"] == "assistant":
            last_assistant_turn = message["content"]
            break
    for index in range(len(chat_engine.history) - 1, -1, -1):
        if chat_engine.history[index]["role"] == "user":
            last_user_turn = chat_engine.history[index]["content"]
            break

    file_exists = MANUAL_RATING_FILE.exists()
    with open(MANUAL_RATING_FILE, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "persona_name",
                    "prompt_method",
                    "user_message",
                    "assistant_response",
                    "persona_consistency",
                    "instruction_following",
                    "response_quality",
                    "comment",
                ]
            )
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                chat_engine.persona.name if chat_engine.persona else "",
                chat_engine.prompt_method,
                last_user_turn or "",
                last_assistant_turn or "",
                int(persona_consistency),
                int(instruction_following),
                int(response_quality),
                comment.strip(),
            ]
        )

    return (
        f"Rating saved to {MANUAL_RATING_FILE.name} "
        f"(consistency={persona_consistency}, instruction={instruction_following}, "
        f"quality={response_quality})."
    )


# ---------------------------------------------------------------------------
# Gradio UI definition
# ---------------------------------------------------------------------------
def build_interface() -> gr.Blocks:
    """Construct and return the Gradio Blocks interface."""
    persona_choices = list(ALL_PERSONAS.keys())
    persona_choice_labels = {key: ALL_PERSONAS[key].name for key in persona_choices}

    with gr.Blocks(
        title="Qwen2.5 Multi-Persona Chatbot",
    ) as demo:
        gr.Markdown(
            """
            # Conversational AI Role Play
            ### Qwen2.5-3B Performance Analysis and Personalization for Multi-Persona Chatbots

            Customise a persona, compare **basic** versus **structured** prompting,
            run multi-turn conversations, and rate responses for the project experiment.
            """
        )

        with gr.Row():
            # ----------------------------------------------------------
            # Left column: persona and generation configuration
            # ----------------------------------------------------------
            with gr.Column(scale=4):
                gr.Markdown("## Persona Configuration")

                predefined_persona = gr.Dropdown(
                    choices=[
                        (label, key) for key, label in persona_choice_labels.items()
                    ],
                    label="Load a predefined persona",
                    value=None,
                )
                load_persona_btn = gr.Button("Load selected persona", size="sm")

                persona_name = gr.Textbox(
                    label="Character name",
                    value="Assistant",
                    placeholder="e.g. Professor Eleanor Whitfield",
                )

                with gr.Tabs():
                    with gr.Tab("Free-text description"):
                        raw_description = gr.Textbox(
                            label="Raw character description",
                            lines=6,
                            placeholder=(
                                "Describe the character's personality, background and "
                                "speaking style in free text ..."
                            ),
                        )
                    with gr.Tab("Structured fields"):
                        personality_traits = gr.Textbox(
                            label="Personality traits (comma-separated)",
                            placeholder="e.g. analytical, patient, formal",
                        )
                        speaking_style = gr.Textbox(
                            label="Speaking style",
                            placeholder="e.g. formal academic tone, precise vocabulary",
                        )
                        background = gr.Textbox(
                            label="Background",
                            placeholder="e.g. PhD in Computer Science, 30 years teaching",
                        )
                        behavioral_traits = gr.Textbox(
                            label="Behavioral rules (comma-separated)",
                            placeholder="e.g. asks for evidence, corrects misconceptions",
                            lines=3,
                        )

                prompt_method = gr.Radio(
                    choices=[
                        ("Basic Prompt", PROMPT_METHOD_BASIC),
                        ("Structured Persona Prompt", PROMPT_METHOD_STRUCTURED),
                    ],
                    value=PROMPT_METHOD_BASIC,
                    label="Prompt strategy",
                )

                clear_history_checkbox = gr.Checkbox(
                    value=True,
                    label="Clear conversation history when applying a new persona",
                )

                apply_persona_btn = gr.Button("Apply persona", variant="primary")
                reset_btn = gr.Button("Reset conversation")

                with gr.Accordion("Generation parameters", open=False):
                    temperature_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.5,
                        value=DEFAULT_GENERATION_CONFIG.temperature,
                        step=0.05,
                        label="Temperature",
                    )
                    top_p_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=DEFAULT_GENERATION_CONFIG.top_p,
                        step=0.05,
                        label="Top-p",
                    )
                    max_tokens_slider = gr.Slider(
                        minimum=64,
                        maximum=1024,
                        value=DEFAULT_GENERATION_CONFIG.max_new_tokens,
                        step=32,
                        label="Max new tokens",
                    )
                    repetition_penalty_slider = gr.Slider(
                        minimum=1.0,
                        maximum=1.3,
                        value=DEFAULT_GENERATION_CONFIG.repetition_penalty,
                        step=0.01,
                        label="Repetition penalty",
                    )

                with gr.Accordion("Preview system prompt", open=False):
                    system_prompt_preview = gr.Textbox(
                        value=chat_engine.get_system_prompt(),
                        lines=14,
                        label="System prompt sent to the model",
                    )

            # ----------------------------------------------------------
            # Right column: chat and scoring
            # ----------------------------------------------------------
            with gr.Column(scale=6):
                gr.Markdown("## Conversation")

                chatbot = gr.Chatbot(
                    value=[],
                    label="Multi-persona chat",
                    height=460,
                )

                response_info = gr.Textbox(
                    label="Last response metadata",
                    interactive=False,
                    placeholder="Response time and token counts will appear here.",
                )

                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder="Type your message and press Enter ...",
                        scale=8,
                        show_label=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=2)

                with gr.Accordion("Rate the latest response (1-5 scale)", open=False):
                    gr.Markdown(
                        f"Rating scale: **{MIN_RATING} = completely breaks character / "
                        f"poor quality**, **{MAX_RATING} = fully consistent / excellent**."
                    )
                    with gr.Row():
                        rating_consistency = gr.Slider(
                            minimum=MIN_RATING,
                            maximum=MAX_RATING,
                            step=1,
                            value=3,
                            label="Persona consistency",
                        )
                        rating_instruction = gr.Slider(
                            minimum=MIN_RATING,
                            maximum=MAX_RATING,
                            step=1,
                            value=3,
                            label="Instruction following",
                        )
                        rating_quality = gr.Slider(
                            minimum=MIN_RATING,
                            maximum=MAX_RATING,
                            step=1,
                            value=3,
                            label="Response quality",
                        )
                    rating_comment = gr.Textbox(
                        label="Optional comment",
                        placeholder="Note any persona failure, coherence issue or success case.",
                    )
                    save_rating_btn = gr.Button("Save rating")
                    rating_status = gr.Textbox(interactive=False, label="Rating status")

                status_message = gr.Markdown("")

        # ------------------------------------------------------------------
        # Event wiring
        # ------------------------------------------------------------------
        load_persona_btn.click(
            fn=load_predefined_persona,
            inputs=[predefined_persona],
            outputs=[
                persona_name,
                raw_description,
                personality_traits,
                speaking_style,
                background,
                behavioral_traits,
            ],
        )

        apply_persona_btn.click(
            fn=apply_persona,
            inputs=[
                persona_name,
                raw_description,
                personality_traits,
                speaking_style,
                background,
                behavioral_traits,
                prompt_method,
                clear_history_checkbox,
            ],
            outputs=[status_message, system_prompt_preview, chatbot],
        )

        reset_btn.click(
            fn=reset_conversation,
            inputs=None,
            outputs=[chatbot, response_info],
        )

        send_btn.click(
            fn=respond,
            inputs=[
                user_input,
                temperature_slider,
                top_p_slider,
                max_tokens_slider,
                repetition_penalty_slider,
            ],
            outputs=[chatbot, user_input, response_info],
        )

        user_input.submit(
            fn=respond,
            inputs=[
                user_input,
                temperature_slider,
                top_p_slider,
                max_tokens_slider,
                repetition_penalty_slider,
            ],
            outputs=[chatbot, user_input, response_info],
        )

        save_rating_btn.click(
            fn=save_manual_rating,
            inputs=[
                rating_consistency,
                rating_instruction,
                rating_quality,
                rating_comment,
            ],
            outputs=[rating_status],
        )

    return demo


def main() -> None:
    """Entry point for launching the Gradio application."""
    demo = build_interface()
    demo.queue()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="blue"),
    )


if __name__ == "__main__":
    main()
