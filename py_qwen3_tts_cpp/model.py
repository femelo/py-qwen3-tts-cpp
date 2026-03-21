#!/usr/bin/env python

from . import _py_qwen3_tts_cpp as native
import importlib.metadata
import logging
import shutil
import subprocess
import os
import tempfile
from pathlib import Path
from collections.abc import Callable
from typing import Any
from dataclasses import dataclass

import numpy as np
import py_qwen3_tts_cpp.utils as utils
from py_qwen3_tts_cpp.constants import QWEN_TTS_SAMPLE_RATE, LANGUAGE_IDS


__author__ = "femelo"
__copyright__ = "Copyright 2026, "
__license__ = "Apache 2.0"
__version__ = importlib.metadata.version("py_qwen3_tts_cpp")

logger = logging.getLogger(__name__)


@dataclass
class Parameters:
    language: str = "en"
    max_audio_tokens: int = 4096
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.05
    n_threads: int = 4
    print_timing: bool = False
    print_progress: bool = False


class Qwen3TTSModel:
    """
    A high-level wrapper for the Qwen3 TTS engine.
    Handles speech synthesis, voice cloning, and model management.
    """

    def __init__(
        self,
        tts_model: str,
        tokenizer_model: str | None = None,
        models_dir: str | None = None,
        language: str = "en",
        max_audio_tokens: int = 4096,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
        n_threads: int = 4,
        print_timing: bool = False,
        print_progress: bool = False,
    ) -> None:
        if Path(tts_model).is_file():
            self.tts_model_path = tts_model
        else:
            self.tts_model_path = utils.download_model(
                tts_model,
                models_dir,
                model_type=utils.ModelType.TTS,
            )

        if tokenizer_model:
            if Path(tokenizer_model).is_file():
                self.tokenizer_model_path = tokenizer_model
            else:
                self.tokenizer_model_path = utils.download_model(
                    tokenizer_model,
                    models_dir,
                    model_type=utils.ModelType.TOKENIZER,
                )
        else:
            self.tokenizer_model_path = None

        self.tts = native.Qwen3TTS()
        self._models_dir: str | None = models_dir
        self._params: Parameters = Parameters(
            language=language,
            max_audio_tokens=max_audio_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            n_threads=n_threads,
            print_timing=print_timing,
            print_progress=print_progress,
        )
        self._progress_callback: Callable[[int, int], None] | None = None

        # Load models
        self.load_tts_models()

    # --- Model Loading ---

    def load_tts_models(self) -> bool:
        """Loads the GGUF TTS and tokenizer models."""
        if self.tokenizer_model_path:
            if not Path(self.tts_model_path).exists():
                raise ValueError(f"TTS model file could not be found: {self.tts_model_path}")
            if not Path(self.tokenizer_model_path).exists():
                raise ValueError(f"Tokenizer model file could not be found: {self.tokenizer_model_path}")
            return self.tts.load_models(self.tts_model_path, self.tokenizer_model_path)
        else:
            model_dir = str(Path(self.tts_model_path).parent)
            return self.tts.load_models_from_dir(model_dir)

    @staticmethod
    def load_audio(media_file_path: str) -> tuple[np.ndarray, int]:
        """
        Helper method to return a (np.array, sample_rate) tuple from a media file.
        If the media file is not a WAV file at 24kHz, it will try to convert it using ffmpeg.

        :param media_file_path: Path of the media file
        :return: Tuple of (numpy array of float32 samples, sample_rate)
        """
        convert: bool = False
        if media_file_path.endswith(".wav"):
            samples, sample_rate = native.load_audio_file(media_file_path)
            if sample_rate != QWEN_TTS_SAMPLE_RATE:
                convert = True
        else:
            if shutil.which("ffmpeg") is None:
                raise Exception(
                    "FFMPEG is not installed or not in PATH. Please install it, or provide a 24kHz WAV file or a NumPy array instead!"
                )
            convert = True

        if convert:
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_file_path = temp_file.name
            temp_file.close()
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        media_file_path,
                        "-ac",
                        "1",
                        "-ar",
                        str(QWEN_TTS_SAMPLE_RATE),
                        temp_file_path,
                        "-y",
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                samples, sample_rate = native.load_audio_file(temp_file_path)
            finally:
                os.remove(temp_file_path)
        return samples, sample_rate

    def _build_params(
        self,
        language: str | None,
        max_audio_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        top_k: int | None,
        repetition_penalty: float | None,
        n_threads: int | None,
    ) -> native.TtsParams:
        params = native.TtsParams()
        lang = language or self._params.language or "en"
        params.language_id = LANGUAGE_IDS.get(lang, LANGUAGE_IDS["en"])
        params.max_audio_tokens = max_audio_tokens or self._params.max_audio_tokens
        params.temperature = temperature if temperature is not None else self._params.temperature
        params.top_p = top_p if top_p is not None else self._params.top_p
        params.top_k = top_k if top_k is not None else self._params.top_k
        params.repetition_penalty = repetition_penalty if repetition_penalty is not None else self._params.repetition_penalty
        params.n_threads = n_threads or self._params.n_threads
        params.print_timing = self._params.print_timing
        params.print_progress = self._params.print_progress
        return params

    # --- Speech Synthesis ---

    def synthesize(
        self,
        text: str,
        language: str | None = None,
        max_audio_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        n_threads: int | None = None,
    ) -> native.TtsResult:
        """
        Synthesizes speech from text.

        :param text: Input text to synthesize
        :return: TtsResult with audio samples and timing info
        """
        params = self._build_params(language, max_audio_tokens, temperature, top_p, top_k, repetition_penalty, n_threads)
        return self.tts.synthesize(text, params)

    def synthesize_with_voice(
        self,
        text: str,
        reference_audio: str | np.ndarray,
        language: str | None = None,
        max_audio_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        n_threads: int | None = None,
    ) -> native.TtsResult:
        """
        Synthesizes speech with voice cloning from a reference audio file or NumPy array.

        :param text: Input text to synthesize
        :param reference_audio: Path to reference audio file (WAV, 24kHz) or NumPy array of samples
        :return: TtsResult with audio samples and timing info
        """
        params = self._build_params(language, max_audio_tokens, temperature, top_p, top_k, repetition_penalty, n_threads)

        if isinstance(reference_audio, str):
            return self.tts.synthesize_with_voice_file(text, reference_audio, params)
        elif isinstance(reference_audio, np.ndarray):
            ref_samples = reference_audio.astype(np.float32)
            return self.tts.synthesize_with_voice_samples(text, ref_samples, params)
        else:
            raise ValueError("reference_audio must be a file path (str) or NumPy array.")

    def extract_speaker_embedding(
        self,
        reference_audio: str | np.ndarray,
        params: native.TtsParams | None = None,
    ) -> np.ndarray:
        """
        Extracts a speaker embedding from a reference audio for reuse across multiple calls.

        :param reference_audio: Path to WAV file or NumPy array (24kHz, mono, float32)
        :return: Speaker embedding as a NumPy array
        """
        if params is None:
            params = self._build_params(None, None, None, None, None, None, None)

        if isinstance(reference_audio, str):
            samples, _ = Qwen3TTSModel.load_audio(reference_audio)
        elif isinstance(reference_audio, np.ndarray):
            samples = reference_audio.astype(np.float32)
        else:
            raise ValueError("reference_audio must be a file path (str) or NumPy array.")

        return self.tts.extract_speaker_embedding(samples, params)

    def synthesize_with_embedding(
        self,
        text: str,
        embedding: np.ndarray,
        language: str | None = None,
        max_audio_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        n_threads: int | None = None,
    ) -> native.TtsResult:
        """
        Synthesizes speech with a pre-computed speaker embedding (skips the encoder step).

        :param text: Input text to synthesize
        :param embedding: Speaker embedding from extract_speaker_embedding()
        :return: TtsResult with audio samples and timing info
        """
        params = self._build_params(language, max_audio_tokens, temperature, top_p, top_k, repetition_penalty, n_threads)
        emb = embedding.astype(np.float32)
        return self.tts.synthesize_with_embedding(text, emb, params)

    def save_audio(self, result: native.TtsResult, output_path: str) -> None:
        """
        Saves a TtsResult audio to a WAV file.

        :param result: TtsResult from any synthesize call
        :param output_path: Path to save the WAV file
        """
        native.save_audio_file(output_path, result.audio, result.sample_rate)

    # --- Utilities & Callbacks ---

    def get_params(self) -> dict[str, Any]:
        """Returns the current model parameters."""
        return self._params.__dict__

    def set_progress_callback(self, callback: Callable[[int, int], None]) -> None:
        """Sets a function to be called during synthesis progress."""
        self._progress_callback = callback
        self.tts.set_progress_callback(callback)

    @property
    def last_error(self) -> str:
        return self.tts.get_error()

    def is_ready(self) -> bool:
        """Checks if the TTS model is loaded and ready."""
        return self.tts.is_loaded()
