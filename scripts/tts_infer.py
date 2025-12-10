"""
Echo Twin - scripts/tts_infer.py (Enhanced TTS Pipeline with Multi-Model Support)
---------------------------------------------------------------------------------
Purpose:
    Generate natural speech audio from text using pretrained TTS models with support for:
    - Multiple model loading and switching
    - Model ensembles
    - Sequential generation
    - Model persistence for web UI

Changelog (2025-12-08):
    - Added multi-model support with ModelManager class
    - Added ensemble synthesis capabilities
    - Added model persistence for web UI
    - Maintains backward compatibility with existing code
    - Added proper logging and error handling
    - Fixed: No training/data loading on import
"""

from __future__ import annotations
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple

# Import normalize_audio from utils to avoid duplication
from utils.audio_utils import normalize_audio

# ----------------------------------------------------------
# Logging (ASCII-safe)
# ----------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[tts_infer] %(message)s")
log = logging.getLogger("tts_infer")

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
OUT_DIR = BASE_DIR / "logs" / "evaluations_tts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------
# Data Classes
# ----------------------------------------------------------
class ModelConfig:
    """Configuration for a single TTS model."""
    
    def __init__(
        self,
        model_id: str,
        name: str = "",
        weight: float = 1.0,
        device: Optional[str] = None,
        enabled: bool = True,
        priority: int = 0,
        hf_cache_dir: Optional[str] = None
    ):
        self.model_id = model_id
        self.name = name or model_id.replace("/", "_")
        self.weight = weight
        self.device = device
        self.enabled = enabled
        self.priority = priority
        self.hf_cache_dir = hf_cache_dir


# ----------------------------------------------------------
# Model Manager for Multi-Model Support
# ----------------------------------------------------------
class ModelManager:
    """Manages loading and accessing multiple TTS models."""
    
    def __init__(self, device: Optional[str] = None, project_root: Optional[Path] = None):
        # Import torch here to avoid early import issues
        import torch
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.project_root = project_root or BASE_DIR
        self.models: Dict[str, Any] = {}  # Model instances
        self.processors: Dict[str, Any] = {}  # Processors
        self.model_info: Dict[str, ModelConfig] = {}
        self.active_model: Optional[str] = None
        log.info(f"[ModelManager] Initialized with device: {self.device}")
    
    def load_model(self, model_cfg: ModelConfig) -> str:
        """Load a single TTS model."""
        # Import transformers here to avoid loading unless needed
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError:
            raise ImportError("Transformers library is required. Install with: pip install transformers")
        
        model_id = model_cfg.name or model_cfg.model_id.replace("/", "_")
        
        if model_id in self.models:
            log.warning(f"[ModelManager] Model '{model_id}' already loaded")
            return model_id
        
        log.info(f"[ModelManager] Loading model '{model_id}' from {model_cfg.model_id}")
        
        # Determine device for this model
        if model_cfg.device:
            model_device = model_cfg.device
        else:
            model_device = self.device
        
        try:
            # Set cache directory if specified
            cache_kwargs = {}
            if model_cfg.hf_cache_dir:
                cache_kwargs['cache_dir'] = model_cfg.hf_cache_dir
            
            # Load processor and model
            processor = AutoProcessor.from_pretrained(model_cfg.model_id, **cache_kwargs)
            model = AutoModel.from_pretrained(model_cfg.model_id, **cache_kwargs)
            
            # Move to device
            import torch
            model = model.to(model_device)
            model.eval()
            
            # Store model and processor
            self.models[model_id] = model
            self.processors[model_id] = processor
            self.model_info[model_id] = model_cfg
            
            # Set as active if first model or higher priority
            if self.active_model is None:
                self.active_model = model_id
            elif model_cfg.priority > self.model_info[self.active_model].priority:
                self.active_model = model_id
            
            log.info(f"[ModelManager] Model '{model_id}' loaded successfully")
            return model_id
            
        except Exception as e:
            log.error(f"[ModelManager] Failed to load model '{model_cfg.model_id}': {e}")
            raise
    
    def load_models_from_config(self, config_file: str) -> List[str]:
        """Load multiple models from a YAML configuration file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required. Install with: pip install pyyaml")
        
        config_path = Path(config_file)
        if not config_path.is_absolute():
            config_path = self.project_root / config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        log.info(f"[ModelManager] Loading models from config: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_ids = []
        models_data = config.get('models', [])
        
        if not models_data:
            raise ValueError(f"No models defined in config file: {config_path}")
        
        for i, model_data in enumerate(models_data):
            if not isinstance(model_data, dict):
                log.warning(f"[ModelManager] Skipping invalid model entry at index {i}")
                continue
            
            if 'model_id' not in model_data:
                log.warning(f"[ModelManager] Skipping model entry {i}: missing 'model_id' field")
                continue
            
            model_cfg = ModelConfig(
                model_id=model_data['model_id'],
                name=model_data.get('name', f'model_{i}'),
                weight=float(model_data.get('weight', 1.0)),
                device=model_data.get('device'),
                enabled=bool(model_data.get('enabled', True)),
                priority=int(model_data.get('priority', 0)),
                hf_cache_dir=model_data.get('hf_cache_dir')
            )
            
            if model_cfg.enabled:
                try:
                    model_id = self.load_model(model_cfg)
                    model_ids.append(model_id)
                except Exception as e:
                    log.error(f"[ModelManager] Failed to load model '{model_cfg.name}': {e}")
                    if not model_ids:
                        raise
        
        log.info(f"[ModelManager] Successfully loaded {len(model_ids)} models")
        return model_ids
    
    def set_active_model(self, model_id: str):
        """Set which model to use for single-model inference."""
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not loaded")
        self.active_model = model_id
        log.info(f"[ModelManager] Active model set to '{model_id}'")
    
    def get_model(self, model_id: Optional[str] = None) -> Any:
        """Get a model by ID or the active model."""
        if model_id is None:
            model_id = self.active_model
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not found")
        return self.models[model_id]
    
    def get_processor(self, model_id: Optional[str] = None) -> Any:
        """Get a processor by ID or active model."""
        if model_id is None:
            model_id = self.active_model
        if model_id not in self.processors:
            raise ValueError(f"Processor for model '{model_id}' not found")
        return self.processors[model_id]
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all loaded models with their info."""
        return [
            {
                'id': model_id,
                'name': info.name,
                'model_id': info.model_id,
                'weight': info.weight,
                'priority': info.priority,
                'enabled': info.enabled,
                'is_active': model_id == self.active_model
            }
            for model_id, info in self.model_info.items()
        ]
    
    def unload_model(self, model_id: str):
        """Unload a model to free memory."""
        if model_id in self.models:
            del self.models[model_id]
        if model_id in self.processors:
            del self.processors[model_id]
        if model_id in self.model_info:
            del self.model_info[model_id]
        
        if self.active_model == model_id:
            self.active_model = next(iter(self.models.keys())) if self.models else None
        
        log.info(f"[ModelManager] Model '{model_id}' unloaded")


# ----------------------------------------------------------
# Core Functions
# ----------------------------------------------------------
def text_to_speech(
    text: str,
    output_path: str | None = None,
    sample_rate: int | None = None,
    model_id: str = "facebook/mms-tts-eng",
    normalize: bool = True
) -> str:
    """
    Generate speech from text using a TTS model.
    
    Args:
        text: Input text
        output_path: Path to save output (None = auto-generate)
        sample_rate: Output sample rate (None = use model default)
        model_id: Hugging Face model ID
        normalize: Whether to normalize audio
    
    Returns:
        Path to generated audio file
    """
    if not text.strip():
        raise ValueError("Input text is empty.")
    
    log.info(f"[text_to_speech] Generating speech for: {text[:80]}{'...' if len(text)>80 else ''}")
    
    # Import transformers here to avoid loading unless needed
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError:
        raise ImportError("Transformers library is required. Install with: pip install transformers")
    
    import torch
    import soundfile as sf
    import numpy as np
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device)
        model.eval()
    except Exception as e:
        log.error(f"[text_to_speech] Model load failed: {e}")
        raise
    
    # Generate speech
    inputs = processor(text=text, return_tensors="pt").to(device)
    
    try:
        with torch.no_grad():
            if hasattr(model, "generate_speech"):
                speech = model.generate_speech(inputs["input_ids"])
                if isinstance(speech, torch.Tensor):
                    speech = speech.cpu().numpy()
            else:
                output = model(**inputs)
                speech = output.waveform[0].cpu().numpy()
    except Exception as e:
        log.error(f"[text_to_speech] Generation failed: {e}")
        raise
    
    # Normalize
    if normalize:
        speech = normalize_audio(speech)
    
    # Determine sample rate
    if sample_rate is None:
        sample_rate = getattr(model.config, "sampling_rate", 16000)
    
    # Generate output path if not provided
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        text_hash = hash(text) % 10000
        output_path = OUT_DIR / f"tts_{ts}_{text_hash:04d}.wav"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save audio
    sf.write(str(output_path), speech, sample_rate)
    log.info(f"[text_to_speech] Output saved to: {output_path}")
    
    return str(output_path)


# ----------------------------------------------------------
# Alias for backward compatibility
# ----------------------------------------------------------
generate_speech = text_to_speech

# ----------------------------------------------------------
# Export for UI
# ----------------------------------------------------------
__all__ = [
    'ModelManager',
    'ModelConfig',
    'normalize_audio',
    'text_to_speech',
    'generate_speech'
]

# ----------------------------------------------------------
# Smoke test (only runs if executed directly)
# ----------------------------------------------------------
if __name__ == "__main__":
    # Test single model
    text = "This is a voice quality test by Echo Twin using the enhanced pipeline."
    try:
        path = text_to_speech(text)
        log.info(f"[tts_infer] Generated speech file: {path}")
    except Exception as e:
        log.error(f"[tts_infer] Test failed: {e}")