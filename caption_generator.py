"""Lazy, cached GIT caption inference. No legacy model is loaded by this module."""

from collections.abc import Callable
import logging
import os
from threading import Lock
from typing import TYPE_CHECKING

from PIL import Image
import streamlit as st

from utils import format_caption

# Prevent Transformers from discovering optional ML backends in mixed local envs.
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

if TYPE_CHECKING:
    import torch
    from transformers import GitForCausalLM, GitProcessor

MODEL_ID = "microsoft/git-base-coco"
MODEL_REVISION = "a13141da42abd4a8cbf283601a8104265f537cee"
NUM_BEAMS = 3
MAX_NEW_TOKENS = 40
NO_REPEAT_NGRAM_SIZE = 2
CPU_THREADS = 2
LOGGER = logging.getLogger(__name__)
# Serialize requests sharing the cached model to limit concurrent beam memory.
INFERENCE_LOCK = Lock()


class CaptionGenerationError(Exception):
    """A model/download/inference failure safe to display to the user."""


def select_device() -> "torch.device":
    """Use CUDA when available; otherwise run safely on CPU."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _raise_friendly_error(error: Exception, message: str) -> None:
    """Translate detectable allocation failures without exposing internal details."""
    if isinstance(error, MemoryError) or any(
        text in str(error).lower() for text in
        ("out of memory", "can't allocate memory", "cannot allocate memory",
         "not enough memory", "defaultcpuallocator")
    ):
        raise CaptionGenerationError(
            "VisionScribe ran out of available memory. Please try again shortly. "
            "If this continues, the host may need more memory."
        ) from error
    raise CaptionGenerationError(message) from error


@st.cache_resource(show_spinner=False)
def load_processor() -> "GitProcessor":
    """Cache the public model's processor; downloads use the Hugging Face cache."""
    try:
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=False,
                                             trust_remote_code=False)
    except Exception as error:
        LOGGER.exception("GIT processor loading failed")
        _raise_friendly_error(
            error, "VisionScribe couldn't load its image processor. Please try again shortly."
        )


@st.cache_resource(show_spinner=False)
def load_caption_model() -> "GitForCausalLM":
    """Load one eval-mode model with reduced loading overhead; never load v1."""
    try:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import GitForCausalLM

        device = select_device()
        if device.type == "cpu":
            torch.set_num_threads(CPU_THREADS)
        # Download only needed files into the Hub cache. Loading that local
        # snapshot avoids Transformers 4.49's background conversion requests.
        snapshot = snapshot_download(
            MODEL_ID, revision=MODEL_REVISION,
            allow_patterns=["config.json", "generation_config.json", "pytorch_model.bin"],
        )
        # PyTorch 2.6 uses restricted weights-only loading for this .bin file.
        model = GitForCausalLM.from_pretrained(
            snapshot, local_files_only=True, low_cpu_mem_usage=True,
            use_safetensors=False, trust_remote_code=False,
        )
        model.to(device)
        model.eval()
        return model
    except Exception as error:
        LOGGER.exception("GIT model loading failed")
        _raise_friendly_error(
            error, "VisionScribe couldn't load its captioning model. Please try again shortly."
        )


def generate_caption(
    image: Image.Image, on_progress: Callable[[str], None] | None = None,
) -> str:
    """Generate a deterministic caption with beam search and special-token decoding.

    The processor owns all model-specific resizing/normalization. Only the small
    input batch is moved to the cached model's device. No upload is cached globally.
    """
    def report(message: str) -> None:
        if on_progress is not None:
            on_progress(message)

    report("Loading VisionScribe's captioning model (first request may take longer)...")
    processor = load_processor()
    model = load_caption_model()
    try:
        import torch

        # Requests wait here instead of allocating concurrent beam-search batches.
        with INFERENCE_LOCK, torch.inference_mode():
            report("Analyzing your image...")
            inputs = processor(images=image, return_tensors="pt").to(model.device)
            report("Generating caption...")
            output_ids = model.generate(
                **inputs, num_beams=NUM_BEAMS, max_new_tokens=MAX_NEW_TOKENS,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE, early_stopping=True,
                do_sample=False,
            )
            decoded = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        caption = format_caption(decoded)
        if not caption:
            raise CaptionGenerationError(
                "VisionScribe couldn't find a description. Please try another photo."
            )
        return caption
    except CaptionGenerationError:
        raise
    except Exception as error:
        LOGGER.exception("GIT caption inference failed")
        _raise_friendly_error(
            error, "VisionScribe couldn't generate a caption. Please try again or choose another image."
        )
