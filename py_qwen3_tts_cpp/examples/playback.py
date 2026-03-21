#!/usr/bin/env python

import argparse
import importlib.metadata
import logging

import sounddevice as sd

import py_qwen3_tts_cpp.constants as constants
from py_qwen3_tts_cpp.model import Qwen3TTSModel


__version__ = importlib.metadata.version("py_qwen3_tts_cpp")

__header__ = f"""
===================================================================
PyQwen3TTS
A simple example of synthesizing and playing back speech
Version: {__version__}
===================================================================
"""


class Playback:
    """
    Playback class — synthesizes text and plays back the result via sounddevice.

    Example usage
    ```python
    from py_qwen3_tts_cpp.examples.playback import Playback

    pb = Playback("Hello, world!")
    pb.start()
    ```
    """

    def __init__(
        self, text: str, model: str = "qwen3-tts-0.6b-f16", **model_params
    ):
        self.text = text
        self.sample_rate = constants.QWEN_TTS_SAMPLE_RATE
        self.pqw3_model = Qwen3TTSModel(tts_model=model, **model_params)

    def start(self):
        logging.info(f"Synthesizing: {self.text!r} ...")
        result = self.pqw3_model.synthesize(self.text)
        if not result.success:
            logging.error(f"Synthesis failed: {result.error_msg}")
            return
        logging.info("Playback starting ...")
        sd.play(result.audio, samplerate=result.sample_rate)
        sd.wait()
        logging.info("Playback finished")


def _main():
    print(__header__)
    parser = argparse.ArgumentParser(description="", allow_abbrev=True)
    parser.add_argument("text", type=str, help="Text to synthesize and play back")
    parser.add_argument(
        "-m",
        "--model",
        default="qwen3-tts-0.6b-f16",
        type=str,
        help="Qwen3 TTS model, default to %(default)s",
    )
    parser.add_argument(
        "-l",
        "--language",
        default="en",
        type=str,
        help=f"Language code, one of {list(constants.LANGUAGE_IDS.keys())}, default to %(default)s",
    )

    args = parser.parse_args()

    pb = Playback(text=args.text, model=args.model, language=args.language)
    pb.start()


if __name__ == "__main__":
    _main()
