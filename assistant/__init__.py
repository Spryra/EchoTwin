# File: ECHO_TWIN_CORE/assistant/__init__.py
# Description: Initialization file for the Echo Twin Assistant package.

from .speech_recognition import transcribe_audio
from .nlp_logic import recognize_intent, generate_response
from .respond import speak_response

__all__ = [
    "transcribe_audio",
    "recognize_intent",
    "generate_response",
    "speak_response",
]
