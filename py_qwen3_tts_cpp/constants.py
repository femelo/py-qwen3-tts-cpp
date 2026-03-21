#!/usr/bin/env python

from pathlib import Path
from platformdirs import user_data_dir


# Constants matching Qwen3-TTS audio output
QWEN_TTS_SAMPLE_RATE = 24000

# Language ID mapping for the codec
LANGUAGE_IDS = {
    "en": 2050,
    "de": 2053,
    "es": 2054,
    "zh": 2055,
    "fr": 2061,
    "ja": 2058,
    "ko": 2064,
    "ru": 2069,
}


# example: "https://huggingface.co/OpenVoiceOS/qwen3-tts-0.6b-f16/resolve/main/qwen3-tts-0.6b-f16.gguf"
MODEL_URL_TEMPLATE = "https://huggingface.co/OpenVoiceOS/{model}/resolve/main/{file_name}.gguf"

PACKAGE_NAME = "py_qwen3_tts_cpp"
MODELS_DIR = Path(user_data_dir(PACKAGE_NAME)) / "models"


AVAILABLE_MODELS = {
    "tts": [
        "qwen3-tts-0.6b-f16",
        "qwen3-tts-0.6b-q8-0",
        "qwen3-tts-0.6b-q5-k-m",
        "qwen3-tts-0.6b-q4-k-m",
    ],
    "tokenizer": [
        "qwen3-tts-tokenizer-f16",
    ],
}


PARAMS_SCHEMA = {
    "tts_model": {
        "type": str,
        "description": "TTS model to use, can be a local path or a model name from the available models list",
        "options": None,
        "default": "qwen3-tts-0.6b-f16",
    },
    "tokenizer_model": {
        "type": str,
        "description": "Tokenizer model to use, can be a local path or a model name from the available models list. "
        "If not specified, it will be automatically determined based on the TTS model.",
        "options": None,
        "default": None,
    },
    "models_dir": {
        "type": str,
        "description": "Directory to store downloaded models, default to platform-specific user data dir",
        "options": None,
        "default": str(MODELS_DIR),
    },
    "n_threads": {
        "type": int,
        "description": "Number of threads to allocate for the inference, "
        "default to min(4, available hardware_concurrency)",
        "options": None,
        "default": 4,
    },
    "max_audio_tokens": {
        "type": int,
        "description": "Maximum audio tokens to generate, the default is 4096.",
        "options": None,
        "default": 4096,
    },
    "temperature": {
        "type": float,
        "description": "Sampling temperature (0 = greedy), default is 0.9.",
        "options": None,
        "default": 0.9,
    },
    "top_p": {
        "type": float,
        "description": "Top-p nucleus sampling probability, default is 1.0.",
        "options": None,
        "default": 1.0,
    },
    "top_k": {
        "type": int,
        "description": "Top-k sampling (0 = disabled), default is 50.",
        "options": None,
        "default": 50,
    },
    "repetition_penalty": {
        "type": float,
        "description": "Repetition penalty for token generation, default is 1.05.",
        "options": None,
        "default": 1.05,
    },
    "language": {
        "type": str,
        "description": f"Language code for synthesis, one of: {list(LANGUAGE_IDS.keys())}. Default is 'en'.",
        "options": list(LANGUAGE_IDS.keys()),
        "default": "en",
    },
    "print_timing": {
        "type": bool,
        "description": "Print timing summary",
        "options": None,
        "default": False,
    },
    "print_progress": {
        "type": bool,
        "description": "Print progress information",
        "options": None,
        "default": False,
    },
}

PARAMS_MAPPING: dict = {}
