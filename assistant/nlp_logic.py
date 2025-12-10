# File: ECHO_TWIN_CORE/assistant/nlp_logic.py
# Description: Simple rule-based NLP engine for recognizing user intent and generating responses.

import re
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("assistant.nlp_logic")

# Basic intent mappings for simple conversation
INTENT_PATTERNS = {
    "greeting": [r"\bhello\b", r"\bhi\b", r"\bhey\b"],
    "time_query": [r"what( is|'s)? the time", r"current time", r"tell me the time"],
    "identity": [r"who (are|r) you", r"what is your name", r"introduce yourself"],
    "goodbye": [r"goodbye", r"bye", r"see you"],
    "demo_echo": [r"say hello", r"echo.*twin"],
}

def recognize_intent(text):
    """Detects user intent based on simple regex matching."""
    text = text.lower().strip()
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                log.info(f"Recognized intent: {intent}")
                return intent
    return "unknown"

def generate_response(intent):
    """Generates a simple response string based on recognized intent."""
    if intent == "greeting":
        return "Hello there! I am Echo Twin, your virtual assistant."
    elif intent == "time_query":
        import datetime
        current_time = datetime.datetime.now().strftime("%H:%M")
        return f"The current time is {current_time}."
    elif intent == "identity":
        return "I am Echo Twin, an AI-based personal assistant built from scratch."
    elif intent == "goodbye":
        return "Goodbye! Have a great day."
    elif intent == "demo_echo":
        return "Hello! This is Echo Twin speaking."
    else:
        return "I'm sorry, I didn’t quite catch that. Could you rephrase it?"
