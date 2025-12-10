"""
Echo Twin - scripts/ui_app.py
Simple TTS Web Interface - No Training Required
"""

from __future__ import annotations
import os
import sys
import logging
import gradio as gr
from pathlib import Path

# Add the parent directory to the Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[ui_app] %(message)s")
log = logging.getLogger("ui_app")

# ---------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------
def main():
    log.info("Launching Echo Twin TTS UI...")
    log.info("Note: TTS models will download automatically on first use (no training needed)")
    
    # Create UI
    with gr.Blocks(title="Echo Twin - Text to Speech", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎤 Echo Twin - Text to Speech")
        gr.Markdown("Enter text and let Echo Twin synthesize speech using Hugging Face TTS models.")
        
        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Enter text to speak",
                    placeholder="Type something here...",
                    lines=4
                )
                generate_btn = gr.Button("🎵 Generate Speech", variant="primary")
                
                # Add model selection
                model_dropdown = gr.Dropdown(
                    choices=["facebook/mms-tts-eng", "facebook/mms-tts-fra", "facebook/mms-tts-spa"],
                    value="facebook/mms-tts-eng",
                    label="TTS Model",
                    info="Select which model to use"
                )
                
            with gr.Column(scale=1):
                audio_output = gr.Audio(
                    label="Generated Speech",
                    type="filepath",
                    autoplay=True
                )
                status_text = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False
                )
        
        # Examples
        examples = [
            "Hello, this is Echo Twin speaking.",
            "The quick brown fox jumps over the lazy dog.",
            "Welcome to the future of text-to-speech technology.",
        ]
        
        gr.Examples(
            examples=examples,
            inputs=[text_input],
            label="Try these examples:"
        )
        
        # Generation function - uses TTS directly, no training needed
        def process_text(text: str, model_id: str = "facebook/mms-tts-eng"):
            """Generate speech from text using TTS model - no training required."""
            if not text.strip():
                return None, "❌ Please enter some text"
            
            try:
                # Import TTS function directly - avoid importing scripts package
                # This prevents train_model.py from loading mel spectrograms
                import importlib.util
                tts_file = Path(__file__).parent / "tts_infer.py"
                spec = importlib.util.spec_from_file_location("_tts_infer", tts_file)
                tts_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(tts_module)
                text_to_speech = tts_module.text_to_speech
                
                # Generate output path
                import hashlib
                from datetime import datetime
                
                text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = Path("outputs/ui_demo")
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"ui_{timestamp}_{text_hash}.wav"
                
                # Generate audio using TTS (model downloads automatically on first use)
                log.info(f"Generating speech with model: {model_id}")
                audio_path = text_to_speech(
                    text=text,
                    output_path=str(output_path),
                    model_id=model_id,
                    normalize=True
                )
                
                if audio_path:
                    return audio_path, f"✅ Audio generated using {model_id}"
                else:
                    return None, "❌ Failed to generate audio"
                    
            except Exception as e:
                log.error(f"Generation error: {e}")
                import traceback
                log.error(traceback.format_exc())
                error_msg = str(e)[:200]
                return None, f"❌ Error: {error_msg}"
        
        # Connect button and text input
        generate_btn.click(
            fn=process_text,
            inputs=[text_input, model_dropdown],
            outputs=[audio_output, status_text]
        )
        
        text_input.submit(
            fn=process_text,
            inputs=[text_input, model_dropdown],
            outputs=[audio_output, status_text]
        )
        
    # Launch
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )

# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()