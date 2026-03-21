#!/usr/bin/env python

"""
A simple Command Line Interface to test the package
"""

import argparse
import importlib.metadata
import logging

import py_qwen3_tts_cpp.constants as constants
from py_qwen3_tts_cpp.model import Qwen3TTSModel

__version__ = importlib.metadata.version("py_qwen3_tts_cpp")

__header__ = f"""
PyQwen3TTS
A simple Command Line Interface to test the package
Version: {__version__}
====================================================
"""

logger = logging.getLogger("py_qwen3_tts_cpp-cli")
logger.setLevel(logging.DEBUG)
logging.basicConfig()


def _get_params(args: argparse.Namespace) -> dict:
    """
    Helper function to get params from argparse as a `dict`
    """
    params = {}
    inv_params_mapping = {v: k for k, v in constants.PARAMS_MAPPING.items()}
    for arg in args.__dict__:
        if arg in constants.PARAMS_SCHEMA and getattr(args, arg) is not None:
            params[arg] = getattr(args, arg)
        elif arg in inv_params_mapping:
            arg_ = inv_params_mapping[arg]
            if "." in arg_:
                arg_, subarg_ = arg_.split(".")
                if arg_ not in params:
                    params[arg_] = {}
                params[arg_][subarg_] = getattr(args, arg)
            else:
                params[arg_] = getattr(args, arg)
        else:
            pass
    return params


def run(args: argparse.Namespace) -> None:
    params = _get_params(args)
    m = Qwen3TTSModel(**params)
    if args.verbose:
        print("\npy-qwen3-tts-cli")
        for key, value in m.get_params().items():
            print(f"  {key}: {value}")
    for text in args.text:
        logger.debug(f"Synthesizing: {text!r} ...")
        try:
            result = m.synthesize(text)
            if not result.success:
                logger.error(f"Synthesis failed: {result.error_msg}")
                continue
            if args.output:
                out_path = args.output
            else:
                # derive output filename from input index
                idx = args.text.index(text)
                out_path = f"output_{idx}.wav" if len(args.text) > 1 else "output.wav"
            m.save_audio(result, out_path)
            logger.info(f"Saved audio to: {out_path}")
        except KeyboardInterrupt:
            logger.info("Synthesis manually stopped")
            break
        except Exception as e:
            logger.error(f"Error while synthesizing: {e}")


def main() -> None:
    print(__header__)
    parser = argparse.ArgumentParser(description="", allow_abbrev=True)
    # Positional args
    parser.add_argument(
        "text",
        type=str,
        nargs="+",
        help="The text to synthesize, or a list of texts separated by space",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output WAV file path (default: output.wav)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Whether to print detailed logs (default: False)",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    # add params from PARAMS_SCHEMA
    for param in constants.PARAMS_SCHEMA:
        param_fields = constants.PARAMS_SCHEMA[param]
        type_ = param_fields["type"]
        descr_ = param_fields["description"]
        default_ = param_fields["default"]
        if type_ is dict:
            for dft_key, dft_val in default_.items():
                map_key = f"{param}.{dft_key}"
                map_val = constants.PARAMS_MAPPING.get(map_key)
                if map_val:
                    parser.add_argument(
                        f"--{map_val.replace('_', '-')}",
                        type=type(dft_val),
                        default=dft_val,
                        help=map_key,
                    )
        elif param in constants.PARAMS_MAPPING:
            mapped_param = constants.PARAMS_MAPPING[param]
            parser.add_argument(
                f"--{mapped_param.replace('_', '-')}",
                type=type_,
                default=default_,
                help=descr_,
            )
        elif type_ is bool:
            parser.add_argument(
                f"--{param.replace('_', '-')}",
                type=lambda v: v.lower() in ("true", "yes", "y", "1"),
                default=default_,
                help=descr_,
            )
        else:
            parser.add_argument(
                f"--{param.replace('_', '-')}",
                type=type_,
                default=default_,
                help=descr_,
            )

    args, _ = parser.parse_known_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    run(args)


if __name__ == "__main__":
    main()
