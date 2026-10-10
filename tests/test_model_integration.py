"""Explicitly enabled real inference; never runs by default or in CI."""

import os
from pathlib import Path
import sys
from time import perf_counter

import pytest

from caption_generator import generate_caption, load_caption_model
from utils import open_uploaded_image


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_MODEL_INTEGRATION") != "1",
                    reason="Set RUN_MODEL_INTEGRATION=1 and CAPTION_TEST_IMAGE to a real local photo.")
def test_real_git_model():
    path = Path(os.environ.get("CAPTION_TEST_IMAGE", ""))
    assert path.is_file(), "CAPTION_TEST_IMAGE must name a real photo; no dataset is downloaded by this test."
    with path.open("rb") as file:
        image = open_uploaded_image(file)
    started = perf_counter()
    first = generate_caption(image)
    first_seconds = perf_counter() - started
    model = load_caption_model()
    assert load_caption_model() is model
    assert not model.training
    started = perf_counter()
    second = generate_caption(image)
    assert first == second
    assert first and first[-1] in ".!?"
    assert "tensorflow" not in sys.modules
    print(f"Caption: {first}\nFirst request: {first_seconds:.2f}s\n"
          f"Cached inference: {perf_counter() - started:.2f}s")
