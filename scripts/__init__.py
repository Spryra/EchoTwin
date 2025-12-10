# File: ECHO_TWIN_CORE/scripts/__init__.py
# Description: Allows direct import of training and evaluation scripts as modules.
# NOTE: No auto-imports to prevent train_model.py from loading mel spectrograms on import.
# Users should import explicitly: from scripts.train_model import ...

# Empty __init__ - all imports must be explicit to avoid triggering training code
__all__ = []
