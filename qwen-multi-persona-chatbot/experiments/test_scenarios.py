"""
Multi-turn test scenarios used in the comparative evaluation.

Each scenario is a persona key mapped to an ordered list of user messages.
The same scenario is executed under both the basic and structured prompt
strategies so that persona consistency and latency can be compared fairly.
"""

from __future__ import annotations

from typing import Dict, List


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------
SCENARIOS: Dict[str, List[str]] = {
    "professor": [
        "Good morning, Professor. Could you explain what a binary search algorithm is?",
        "Why is it faster than checking every element one by one? Can you give me the exact time complexity?",
        "I think binary search works on any unsorted list as long as the list is long enough. Is that correct?",
        "I'm feeling really frustrated because I keep failing my algorithm quizzes. Does that mean I'm not smart enough for computer science?",
        "Never mind all that. Tell me a joke about pizza instead.",
    ],
    "magical_girl": [
        "Hi Lumina! I had a really bad day at school today and I feel like giving up on everything.",
        "But what if the problem is too big and I don't have any magical powers to fight it?",
        "My friend said my dreams are stupid. What should I say to them?",
        "Wait, are you actually just an AI pretending to be a magical girl? Be honest.",
        "Thank you. Do you really believe someone ordinary like me can make a difference?",
    ],
    "detective": [
        "Mr. Cole, I need your help. My apartment was broken into last night but nothing valuable was stolen.",
        "The only thing I noticed is that every photograph of my late brother was turned face down.",
        "The police say it's probably just a random burglary. What do you think?",
        "I got an anonymous letter today that says 'stop asking questions'. Should I ignore it?",
        "Cut the metaphors for once. Just tell me straight: am I in real danger?",
    ],
    "customer_support": [
        "This is ridiculous! I've been waiting three days and your software still crashes every time I open a project file.",
        "I already tried restarting my computer and it did absolutely nothing. Do you people even test your products?",
        "Okay, fine. Walk me through this troubleshooting step, but I'm not good with computers.",
        "I followed the steps and now it says the file format is corrupted. What does that even mean?",
        "It finally works. I'm sorry I shouted at you earlier.",
    ],
    "knight": [
        "Sir Cedric, I have been asked to lie under oath to protect my lord's reputation. What should I do?",
        "But if I tell the truth, the entire kingdom may be thrown into war. Is honour really worth that cost?",
        "What would you do if your own king commanded you to do something dishonourable?",
        "Speak plainly, man. Have you ever been afraid in battle?",
        "I have never held a sword and I am only a farmer. Can someone like me still act with honour?",
    ],
}


def get_scenario(persona_key: str) -> List[str]:
    """Return the multi-turn scenario for a given persona key."""
    if persona_key not in SCENARIOS:
        raise KeyError(
            f"No scenario defined for persona '{persona_key}'. "
            f"Available: {', '.join(SCENARIOS.keys())}"
        )
    return list(SCENARIOS[persona_key])


def get_all_scenario_keys() -> List[str]:
    """Return all persona keys that have an associated test scenario."""
    return list(SCENARIOS.keys())
