# assistant/respond.py
"""
Response module: given text reply, synthesize using TTS system and play or save.
"""

import os
import logging
from pathlib import Path
import json
import sounddevice as sd
import soundfile as sf

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("assistant.respond")

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = os.path.join(BASE_DIR, "config", "settings.json")
with open(CONFIG_PATH, "r") as fh:
    SETTINGS = json.load(fh)

def speak_response(text, speaker_wav=None, play=True, outfile=None):
    """
    Synthesize text with TTS system and optionally play or save the output.
    
    Args:
        text: Text to synthesize
        speaker_wav: Reserved for future use (not currently used)
        play: Whether to play audio immediately
        outfile: Output file path (None = auto-generate)
    
    Returns:
        Path to generated audio file
    """
    # Import TTS functions
    import sys
    sys.path.insert(0, str(BASE_DIR))
    
    # Generate output path if not provided
    if outfile is None:
        output_dir = BASE_DIR / "outputs" / "assistant_responses"
        output_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outfile = output_dir / f"assistant_response_{timestamp}.wav"
    
    out_path = str(outfile)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    try:
        # Use regular TTS
        from scripts.tts_infer import text_to_speech
        
        # Get default model from config if available
        model_id = SETTINGS.get("assistant", {}).get("tts_model", "facebook/mms-tts-eng")
        
        out_path = text_to_speech(
            text=text,
            output_path=out_path,
            model_id=model_id,
            normalize=True
        )
        
        # Play audio if requested
        if play:
            try:
                data, sr = sf.read(out_path)
                sd.play(data, sr)
                sd.wait()
                log.info(f"Played audio: {out_path}")
            except Exception as e:
                log.warning(f"Failed to play audio: {e}")
        
        log.info(f"Generated audio: {out_path}")
        return out_path
        
    except Exception as e:
        log.error(f"Failed to synthesize speech: {e}")
        import traceback
        log.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    # Demo interactive loop via microphone file input (for testing)
    demo = speak_response("Hello, this is a quick Echo Twin test.", play=False)
    print("Generated:", demo)
