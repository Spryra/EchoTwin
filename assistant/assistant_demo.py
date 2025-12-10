"""
Echo Twin Assistant - Complete Demo Script
==========================================
This script demonstrates the full assistant workflow:
1. Speech Recognition (Audio → Text)
2. Intent Recognition (Text → Intent)
3. Response Generation (Intent → Response Text)
4. Speech Synthesis (Text → Audio)

Usage:
    python -m assistant.assistant_demo [audio_file.wav]
    
Or run interactively:
    python -m assistant.assistant_demo
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from assistant import transcribe_audio, recognize_intent, generate_response, speak_response

logging.basicConfig(level=logging.INFO, format="[assistant_demo] %(message)s")
log = logging.getLogger("assistant_demo")


def run_assistant_workflow(audio_input_path: str):
    """
    Run the complete assistant workflow.
    
    Args:
        audio_input_path: Path to input audio file (user's speech)
    """
    log.info("=" * 60)
    log.info("Echo Twin Assistant - Complete Workflow")
    log.info("=" * 60)
    
    # Step 1: Speech Recognition (Audio → Text)
    log.info("\n[Step 1] Speech Recognition: Converting audio to text...")
    if not os.path.exists(audio_input_path):
        log.error(f"Audio file not found: {audio_input_path}")
        return
    
    user_text = transcribe_audio(audio_input_path)
    
    if not user_text.strip():
        log.warning("No speech detected in audio file.")
        return
    
    log.info(f"✓ Transcribed text: '{user_text}'")
    
    # Step 2: Intent Recognition (Text → Intent)
    log.info("\n[Step 2] Intent Recognition: Analyzing user intent...")
    intent = recognize_intent(user_text)
    log.info(f"✓ Recognized intent: '{intent}'")
    
    # Step 3: Response Generation (Intent → Response Text)
    log.info("\n[Step 3] Response Generation: Generating response...")
    response_text = generate_response(intent)
    log.info(f"✓ Generated response: '{response_text}'")
    
    # Step 4: Speech Synthesis (Text → Audio)
    log.info("\n[Step 4] Speech Synthesis: Converting response to speech...")
    
    output_dir = BASE_DIR / "outputs" / "assistant_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "assistant_response.wav"
    
    audio_path = speak_response(
        text=response_text,
        play=False,  # Don't auto-play in demo
        outfile=str(output_path)
    )
    
    log.info(f"✓ Generated audio: {audio_path}")
    
    log.info("\n" + "=" * 60)
    log.info("Assistant workflow completed successfully!")
    log.info("=" * 60)
    log.info(f"\nInput audio: {audio_input_path}")
    log.info(f"User said: '{user_text}'")
    log.info(f"Intent: {intent}")
    log.info(f"Response: '{response_text}'")
    log.info(f"Output audio: {audio_path}")
    log.info("\nYou can play the output audio file to hear the assistant's response.")


def interactive_demo():
    """Run assistant in interactive mode (text input)."""
    log.info("=" * 60)
    log.info("Echo Twin Assistant - Interactive Mode")
    log.info("=" * 60)
    log.info("Enter text to interact with the assistant.")
    log.info("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                log.info("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Process through assistant pipeline
            intent = recognize_intent(user_input)
            response_text = generate_response(intent)
            
            print(f"Assistant: {response_text}")
            
            # Optionally generate speech
            generate_speech = input("Generate speech? (y/n): ").strip().lower()
            if generate_speech == 'y':
                output_path = BASE_DIR / "outputs" / "assistant_demo" / "interactive_response.wav"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                audio_path = speak_response(
                    text=response_text,
                    play=True,  # Auto-play in interactive mode
                    outfile=str(output_path),
                )
                print(f"Audio saved to: {audio_path}\n")
            
        except KeyboardInterrupt:
            log.info("\nGoodbye!")
            break
        except Exception as e:
            log.error(f"Error: {e}")
            import traceback
            log.error(traceback.format_exc())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Echo Twin Assistant Demo")
    parser.add_argument(
        "audio_file",
        nargs="?",
        help="Path to input audio file (WAV format). If not provided, runs in interactive text mode."
    )
    
    args = parser.parse_args()
    
    if args.audio_file:
        # Run with audio file
        run_assistant_workflow(audio_input_path=args.audio_file)
    else:
        # Run in interactive mode
        interactive_demo()

