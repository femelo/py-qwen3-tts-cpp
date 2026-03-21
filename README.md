# py-qwen3-tts-cpp

Python bindings for [qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp) using [pybind11](https://github.com/pybind/pybind11).

## Features

- Speech synthesis from text (TTS)
- Voice cloning from a reference WAV file or NumPy array
- Speaker embedding extraction and reuse (cache embeddings to skip the encoder)
- Multi-language support: English, German, Spanish, Chinese, French, Japanese, Korean, Russian
- High-level Python wrapper (`Qwen3TTSModel`) and low-level native bindings
- CLI entry point (`pqw3tts`)

## Installation

### From source

```bash
git clone --recursive https://github.com/femelo/py-qwen3-tts-cpp.git
cd py-qwen3-tts-cpp
pip install .
```

> **Note:** Requires CMake ≥ 3.14, a C++17 compiler, and Python ≥ 3.11.

## Quick start

### Option A — auto-download from HuggingFace

Pass a model name and both the TTS and tokenizer GGUF files are downloaded automatically from the `OpenVoiceOS` HuggingFace repo:

```python
from py_qwen3_tts_cpp.model import Qwen3TTSModel

# Downloads qwen3-tts-0.6b-q8-0.gguf and qwen3-tts-tokenizer-0.6b-q8-0.gguf
model = Qwen3TTSModel(tts_model="qwen3-tts-0.6b-q8-0")

result = model.synthesize("Hello, world!", language="en")
model.save_audio(result, "output.wav")
```

Available model names: `qwen3-tts-0.6b-f16`, `qwen3-tts-0.6b-q8-0`, `qwen3-tts-0.6b-q5-k-m`, `qwen3-tts-0.6b-q4-k-m`.

### Option B — local GGUF files

Pass explicit paths to both GGUF files:

```python
from py_qwen3_tts_cpp.model import Qwen3TTSModel

model = Qwen3TTSModel(
    tts_model="/path/to/models/qwen3-tts-0.6b-f16.gguf",
    tokenizer_model="/path/to/models/qwen3-tts-tokenizer-0.6b-f16.gguf",
)

result = model.synthesize("Hello, world!", language="en")
model.save_audio(result, "output.wav")
```

### Voice cloning

```python
result = model.synthesize_with_voice(
    "Hello, this is a cloned voice.",
    reference_audio="/path/to/reference.wav",
)
model.save_audio(result, "cloned.wav")
```

### Pre-computed speaker embedding

```python
import numpy as np

# Extract once and cache
embedding = model.extract_speaker_embedding("/path/to/reference.wav")
np.save("speaker.npy", embedding)

# Reuse later (skips the encoder)
embedding = np.load("speaker.npy")
result = model.synthesize_with_embedding("Reusing the cached voice.", embedding)
model.save_audio(result, "from_embedding.wav")
```

## CLI

```bash
# Basic synthesis
pqw3tts "Hello, world!" --output hello.wav

# With language and model options
pqw3tts "Hola mundo" --language es --tts-model qwen3-tts-0.6b-f16 --output hola.wav

# Playback example (requires sounddevice)
pqw3tts-playback "Hello from the terminal"
```

Run `pqw3tts --help` for a full list of options.

## Supported languages

| Code | Language   | Language ID |
|------|------------|-------------|
| `en` | English    | 2050        |
| `de` | German     | 2053        |
| `es` | Spanish    | 2054        |
| `zh` | Chinese    | 2055        |
| `fr` | French     | 2061        |
| `ja` | Japanese   | 2058        |
| `ko` | Korean     | 2064        |
| `ru` | Russian    | 2069        |

## API reference

### `Qwen3TTSModel`

| Method | Description |
|---|---|
| `synthesize(text, ...)` | Generate speech from text |
| `synthesize_with_voice(text, reference_audio, ...)` | Voice cloning from file or NumPy array |
| `extract_speaker_embedding(reference_audio)` | Extract speaker embedding for reuse |
| `synthesize_with_embedding(text, embedding, ...)` | Synthesize with cached embedding |
| `save_audio(result, path)` | Save a `TtsResult` to a WAV file |
| `is_ready()` | Check if models are loaded |
| `last_error` | Last error message from the engine |

### `TtsParams` (native)

| Field | Default | Description |
|---|---|---|
| `max_audio_tokens` | 4096 | Maximum audio tokens to generate |
| `temperature` | 0.9 | Sampling temperature |
| `top_p` | 1.0 | Top-p nucleus sampling |
| `top_k` | 50 | Top-k sampling (0 = disabled) |
| `repetition_penalty` | 1.05 | Repetition penalty |
| `n_threads` | 4 | Number of CPU threads |
| `language_id` | 2050 | Language ID (see table above) |
| `print_progress` | False | Print token generation progress |
| `print_timing` | False | Print timing summary |

### `TtsResult` (native)

| Field | Description |
|---|---|
| `audio` | Generated audio as a `numpy.ndarray` (float32, 24 kHz mono) |
| `sample_rate` | Sample rate (always 24000) |
| `success` | Whether synthesis succeeded |
| `error_msg` | Error message on failure |
| `t_load_ms` | Model load time (ms) |
| `t_tokenize_ms` | Tokenization time (ms) |
| `t_encode_ms` | Audio encoding time (ms) |
| `t_generate_ms` | Token generation time (ms) |
| `t_decode_ms` | Audio decoding time (ms) |
| `t_total_ms` | Total synthesis time (ms) |

## License

MIT
