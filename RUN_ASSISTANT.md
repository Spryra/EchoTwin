# How to Run the Echo Twin Assistant

## 🚀 Quick Start - No Training Required!

The assistant works **immediately** without any training. It uses:
- **Pre-trained TTS models** (downloads automatically)
- **Pre-trained ASR models** (Vosk or Whisper)
- **Rule-based NLP** (no training needed)

---

## 📍 Three Ways to Use the Assistant

### **1. Interactive Text Mode (Easiest - No Audio Needed)**

**Command:**
```bash
python -m assistant.assistant_demo
```

**What happens:**
- Prompts you to type text
- Shows assistant's response
- Optionally generates speech

**Example:**
```
You: hello
Assistant: Hello there! I am Echo Twin, your virtual assistant.
Generate speech? (y/n): y
```

**No audio files needed!** Just type and interact.

---

### **2. With Audio File (Full Pipeline)**

**Command:**
```bash
python -m assistant.assistant_demo audio/001.wav
```

**What happens:**
1. Transcribes audio → text
2. Recognizes intent
3. Generates response
4. Synthesizes speech

**Requirements:**
- Audio file with speech (WAV format)
- Vosk model OR Whisper installed

**Output:**
- Console shows all steps
- Audio file: `outputs/assistant_demo/assistant_response.wav`

---

### **3. Quick Test (Intent Recognition Only)**

**Command:**
```bash
python test_assistant.py
```

**What happens:**
- Tests all intents
- Shows responses
- Generates sample speech

**No dependencies needed!** Just runs and shows results.

---

## 🎯 Supported Intents

Try these phrases (in text or audio):

| Phrase | Intent | Response |
|--------|--------|----------|
| "hello" | greeting | "Hello there! I am Echo Twin..." |
| "what time is it" | time_query | "The current time is [time]." |
| "who are you" | identity | "I am Echo Twin, an AI-based..." |
| "goodbye" | goodbye | "Goodbye! Have a great day." |
| "say hello" | demo_echo | "Hello! This is Echo Twin speaking." |

---

## ⚙️ Configuration

### **Speech Recognition Engine**

Edit `config/settings.json`:

```json
{
  "assistant": {
    "asr": "vosk",  // or "whisper"
    "vosk_model_path": "assistant/vosk-model-small"
  }
}
```

**Vosk (Offline):**
- Download from: https://alphacephei.com/vosk/models
- Extract to: `assistant/vosk-model-small/`
- Works offline, fast

**Whisper (Online):**
- Install: `pip install openai-whisper`
- No model download needed
- More accurate

---

## 🎤 Quick Test Right Now

**Run this command:**
```bash
python -m assistant.assistant_demo
```

Then type:
- `hello`
- `what time is it`
- `quit`

**That's it!** The assistant works immediately.

---

## 📁 Output Files

- **Assistant responses:** `outputs/assistant_demo/assistant_response.wav`
- **Interactive mode:** `outputs/assistant_demo/interactive_response.wav`
- **Test outputs:** `outputs/assistant_test_response.wav`

---

## ✅ What Works Without Training

✅ **Intent Recognition** - Rule-based, works immediately
✅ **Response Generation** - Pre-defined responses, works immediately  
✅ **Speech Synthesis** - Uses pre-trained TTS models, downloads automatically
✅ **Interactive Mode** - Works with just text input

---

## ⚠️ What Needs Setup

**Speech Recognition (for audio input):**
- Vosk: Download model and set path in config
- Whisper: Install with `pip install openai-whisper`

**TTS Models:**
- Downloads automatically on first use
- No manual setup needed

---

## 🎉 Ready to Use!

The assistant is **ready to run right now**. Start with interactive mode - it's the easiest!

```bash
python -m assistant.assistant_demo
```

