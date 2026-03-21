# Py-Qwen3-TTS-cpp

Python bindings for [qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp) using [pybind11](https://github.com/pybind/pybind11). Powered by a high-performance C++ backend. This library provides a seamless way to synthesize speech and clone voices using GGUF models.

## 🚀 Features

* **High-Level Wrapper**: Simple, pythonic API for speech synthesis and voice cloning.
* **Automatic Model Download**: Pass a HuggingFace repo ID and both GGUF files are fetched automatically.
* **Voice Cloning**: Clone any voice from a reference WAV file or a NumPy array.
* **Speaker Embedding Cache**: Extract embeddings once and reuse them to skip the encoder on subsequent calls.
* **NumPy Integration**: Receive generated audio directly as `np.float32` arrays.
* **Multi-Language Support**: English, German, Spanish, Chinese, French, Japanese, Korean, Russian.

---

## 📦 Installation

```bash
pip install py-qwen3-tts-cpp
```

**Note**: For non-WAV reference audio files, ensure `ffmpeg` is installed and available in your system `PATH`.

---

## 🛠 Usage

### 1. Basic Synthesis

Synthesize speech from text with just a few lines of code. Pass a HuggingFace repo ID and both GGUF files are downloaded automatically:

```python
from py_qwen3_tts_cpp.model import Qwen3TTSModel

# Downloads qwen3-tts-0.6b-q8-0.gguf and qwen3-tts-tokenizer-0.6b-q8-0.gguf automatically
model = Qwen3TTSModel(tts_model="qwen3-tts-0.6b-q8-0", n_threads=4)

result = model.synthesize("Hello, world!", language="en")
model.save_audio(result, "output.wav")
```

Available model IDs: `qwen3-tts-0.6b-f16`, `qwen3-tts-0.6b-q8-0`, `qwen3-tts-0.6b-q5-k-m`, `qwen3-tts-0.6b-q4-k-m`.

### 2. Local GGUF Files

Pass explicit paths to both GGUF files if you already have them on disk:

```python
model = Qwen3TTSModel(
    tts_model="/path/to/models/qwen3-tts-0.6b-f16.gguf",
    tokenizer_model="/path/to/models/qwen3-tts-tokenizer-0.6b-f16.gguf",
)

result = model.synthesize("Hello, world!", language="en")
model.save_audio(result, "output.wav")
```

### 3. Voice Cloning

Clone a voice from a reference audio file (WAV at 24 kHz recommended, or any format if `ffmpeg` is available):

```python
result = model.synthesize_with_voice(
    "Hello, this is a cloned voice.",
    reference_audio="/path/to/reference.wav",
)
model.save_audio(result, "cloned.wav")
```

### 4. Speaker Embedding Cache

Extract a speaker embedding once and reuse it across multiple calls, skipping the encoder entirely:

```python
import numpy as np

# Extract once and save
embedding = model.extract_speaker_embedding("/path/to/reference.wav")
np.save("speaker.npy", embedding)

# Reuse later (encoder step is skipped)
embedding = np.load("speaker.npy")
result = model.synthesize_with_embedding("Reusing the cached voice.", embedding)
model.save_audio(result, "from_embedding.wav")
```

---

## ⚙️ Configuration

The `Qwen3TTSModel` accepts several parameters to fine-tune performance:

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `tts_model` | `str` | HuggingFace repo ID or path to TTS GGUF file. |
| `tokenizer_model` | `str` | Path to tokenizer GGUF file (auto-downloaded when using a repo ID). |
| `models_dir` | `str` | Directory to cache downloaded models (default: platform user-data dir). |
| `language` | `str` | Language code for synthesis, e.g. `"en"`, `"zh"` (default: `"en"`). |
| `n_threads` | `int` | Number of CPU threads to use (default: 4). |
| `max_audio_tokens` | `int` | Maximum audio tokens to generate (default: 4096). |
| `temperature` | `float` | Sampling temperature — 0 for greedy (default: 0.9). |
| `top_p` | `float` | Top-p nucleus sampling probability (default: 1.0). |
| `top_k` | `int` | Top-k sampling, 0 to disable (default: 50). |
| `repetition_penalty` | `float` | Repetition penalty for token generation (default: 1.05). |
| `print_timing` | `bool` | Print inference timing summary (default: False). |

---

## 🌐 Supported Languages

| Code | Language | Language ID |
| :--- | :--- | :--- |
| `en` | English | 2050 |
| `de` | German | 2053 |
| `es` | Spanish | 2054 |
| `zh` | Chinese | 2055 |
| `fr` | French | 2061 |
| `ja` | Japanese | 2058 |
| `ko` | Korean | 2064 |
| `ru` | Russian | 2069 |

---

## 💻 CLI

```bash
# Basic synthesis
pqw3tts "Hello, world!" --output hello.wav

# With language and model options
pqw3tts "Hola mundo" --language es --tts-model qwen3-tts-0.6b-f16 --output hola.wav

# Synthesize and play back immediately (requires sounddevice)
pqw3tts-playback "Hello from the terminal"
```

Run `pqw3tts --help` for a full list of options.

---

## 📝 License

This project is licensed under the **Apache License 2.0**.

**Author:** femelo
**Copyright:** © 2026

---

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
