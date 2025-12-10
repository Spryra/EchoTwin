"""
Quick test script to demonstrate the Echo Twin Assistant
This shows the assistant working without needing audio files
"""

import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from assistant import recognize_intent, generate_response, speak_response

print("=" * 70)
print("Echo Twin Assistant - Quick Test")
print("=" * 70)
print()

# Test different intents
test_phrases = [
    "hello",
    "what time is it",
    "who are you",
    "goodbye",
    "say hello"
]

print("Testing Intent Recognition and Response Generation:")
print("-" * 70)

for phrase in test_phrases:
    print(f"\nUser says: '{phrase}'")
    
    # Step 1: Recognize intent
    intent = recognize_intent(phrase)
    print(f"  → Intent recognized: {intent}")
    
    # Step 2: Generate response
    response = generate_response(intent)
    print(f"  → Assistant responds: '{response}'")

print("\n" + "=" * 70)
print("Testing Speech Synthesis:")
print("-" * 70)

# Test speech synthesis
test_response = "Hello! I am Echo Twin, your virtual assistant. How can I help you today?"
print(f"\nGenerating speech for: '{test_response}'")

try:
    output_path = speak_response(
        text=test_response,
        play=False,  # Don't auto-play
        outfile="outputs/assistant_test_response.wav"
    )
    print(f"✓ Speech generated successfully!")
    print(f"  Audio saved to: {output_path}")
    print(f"\nYou can play this file to hear the assistant speak.")
except Exception as e:
    print(f"✗ Speech generation failed: {e}")
    print("  Note: This requires the TTS model to be downloaded first.")
    print("  The model will download automatically on first use.")

print("\n" + "=" * 70)
print("Assistant Test Complete!")
print("=" * 70)
print("\nTo test with audio files, run:")
print("  python -m assistant.assistant_demo audio/001.wav")
print("\nFor interactive mode, run:")
print("  python -m assistant.assistant_demo")


