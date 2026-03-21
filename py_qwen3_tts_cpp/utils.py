#!/usr/bin/env python

import contextlib
import logging
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

import requests
from tqdm import tqdm

from py_qwen3_tts_cpp.constants import (
    AVAILABLE_MODELS,
    HF_FILE_URL_TEMPLATE,
    MODELS_DIR,
)


logger = logging.getLogger(__name__)


def _tokenizer_name(tts_model_name: str) -> str:
    """
    Derives the tokenizer file name from the TTS model name.
    e.g. "qwen3-tts-0.6b-q8-0" -> "qwen3-tts-tokenizer-0.6b-q8-0"
    """
    return tts_model_name.replace("qwen3-tts-", "qwen3-tts-tokenizer-", 1)


def _get_tts_urls(tts_model_name: str) -> tuple[str, str]:
    """
    Returns (tts_url, tokenizer_url) for a given TTS model name.
    Both files live in the same HuggingFace repo named after the TTS model.
    """
    tokenizer_name = _tokenizer_name(tts_model_name)
    tts_url = HF_FILE_URL_TEMPLATE.format(repo=tts_model_name, file_name=tts_model_name)
    tokenizer_url = HF_FILE_URL_TEMPLATE.format(repo=tts_model_name, file_name=tokenizer_name)
    return tts_url, tokenizer_url


def _download_file(url: str, file_path: Path, desc: str, chunk_size: int = 1024) -> str:
    """Downloads a single file with a progress bar. Returns the absolute path."""
    if file_path.exists():
        logger.info(f"{desc} already exists at {file_path}")
        return str(file_path.absolute())

    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))

    progress_bar = tqdm(
        desc=f"Downloading {desc} ...",
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    )

    try:
        with open(file_path, "wb") as file, progress_bar:
            for data in resp.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                progress_bar.update(size)
        logger.info(f"Downloaded to {file_path.absolute()}")
    except Exception as e:
        os.remove(file_path)
        raise e

    return str(file_path.absolute())


def download_tts_models(
    tts_model_name: str,
    download_dir: str | None = None,
    chunk_size: int = 1024,
) -> tuple[str, str]:
    """
    Downloads both the TTS and tokenizer GGUF files for the given model name.
    Both files are fetched from the same HuggingFace repo.

    :param tts_model_name: one of AVAILABLE_MODELS, e.g. "qwen3-tts-0.6b-q8-0"
    :param download_dir: local directory to store the files (defaults to platform user-data dir)
    :param chunk_size: download chunk size in bytes
    :return: (tts_model_path, tokenizer_model_path) as absolute path strings
    """
    if tts_model_name not in AVAILABLE_MODELS:
        logger.error(
            f"Invalid model name `{tts_model_name}`, available models are: {AVAILABLE_MODELS}"
        )
        raise ValueError("Invalid model name")

    if download_dir is None:
        download_dir = str(MODELS_DIR)
        logger.info(
            f"No download directory was provided, models will be downloaded to {download_dir}"
        )

    os.makedirs(download_dir, exist_ok=True)

    tts_url, tokenizer_url = _get_tts_urls(tts_model_name)
    tts_path = Path(download_dir) / f"{tts_model_name}.gguf"
    tokenizer_name = _tokenizer_name(tts_model_name)
    tokenizer_path = Path(download_dir) / f"{tokenizer_name}.gguf"

    tts_abs = _download_file(tts_url, tts_path, tts_model_name, chunk_size)
    tokenizer_abs = _download_file(tokenizer_url, tokenizer_path, tokenizer_name, chunk_size)

    return tts_abs, tokenizer_abs


@contextlib.contextmanager
def redirect_stderr(to: bool | TextIO | str | None = False) -> Iterator:
    """
    Redirect stderr to the specified target.

    :param to:
        - None to suppress output (redirect to devnull),
        - sys.stdout to redirect to stdout,
        - A file path (str) to redirect to a file,
        - False to do nothing (no redirection).
    """

    if to is False:
        # do nothing
        yield
        return

    def _resolve_target(target):
        opened_stream = None
        if target is None:
            opened_stream = open(os.devnull, "w")
            return opened_stream, True
        if isinstance(target, str):
            opened_stream = open(target, "w")
            return opened_stream, True
        if hasattr(target, "write"):
            return target, False
        raise ValueError(
            "Invalid `to` parameter; expected None, a filepath string, or a file-like object."
        )

    sys.stderr.flush()
    try:
        original_fd = sys.stderr.fileno()
    except (AttributeError, OSError):
        # Jupyter or non-standard stderr implementations
        original_fd = None

    stream, should_close = _resolve_target(to)

    if original_fd is not None and isinstance(stream, TextIO):
        saved_fd = os.dup(original_fd)
        try:
            os.dup2(stream.fileno(), original_fd)
            yield
        finally:
            os.dup2(saved_fd, original_fd)
            os.close(saved_fd)
            if should_close:
                stream.close()
        return

    # Fallback: Python-level redirect
    if isinstance(stream, TextIO):
        try:
            with contextlib.redirect_stderr(stream):
                yield
        finally:
            if should_close:
                stream.close()
