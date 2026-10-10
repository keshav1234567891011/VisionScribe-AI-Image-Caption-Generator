"""Offline unit tests: all pretrained loaders are mocked."""

import sys
from unittest.mock import Mock, patch

from PIL import Image
import pytest
import torch

import caption_generator as generator


@pytest.fixture(autouse=True)
def reset_caches():
    generator.load_caption_model.clear()
    generator.load_processor.clear()
    yield
    generator.load_caption_model.clear()
    generator.load_processor.clear()


def test_device_selection():
    with patch("torch.cuda.is_available", return_value=False):
        assert generator.select_device().type == "cpu"
    with patch("torch.cuda.is_available", return_value=True):
        assert generator.select_device().type == "cuda"


def test_model_and_processor_load_once_and_eval():
    from transformers import AutoProcessor, GitForCausalLM

    model, processor = Mock(), Mock()
    with patch.object(GitForCausalLM, "from_pretrained", return_value=model) as load, \
         patch.object(AutoProcessor, "from_pretrained", return_value=processor) as process, \
         patch("huggingface_hub.snapshot_download", return_value="cached-snapshot") as snapshot, \
         patch.object(generator, "select_device", return_value=torch.device("cpu")):
        assert generator.load_caption_model() is generator.load_caption_model()
        assert generator.load_processor() is generator.load_processor()
        snapshot.assert_called_once_with(generator.MODEL_ID, revision=generator.MODEL_REVISION,
                                          allow_patterns=["config.json", "generation_config.json", "pytorch_model.bin"])
        load.assert_called_once_with("cached-snapshot", local_files_only=True, low_cpu_mem_usage=True,
                                     use_safetensors=False, trust_remote_code=False)
        process.assert_called_once_with(generator.MODEL_ID, revision=generator.MODEL_REVISION,
                                        use_fast=False, trust_remote_code=False)
        model.eval.assert_called_once()
        model.to.assert_called_once_with(torch.device("cpu"))


@pytest.mark.parametrize("class_name,loader,message", [
    ("GitForCausalLM", generator.load_caption_model, "captioning model"),
    ("AutoProcessor", generator.load_processor, "image processor"),
])
def test_loading_failures_are_friendly(class_name, loader, message):
    import transformers

    with patch.object(getattr(transformers, class_name), "from_pretrained",
                      side_effect=OSError("private/internal/path")), \
         patch("huggingface_hub.snapshot_download", return_value="cached-snapshot"):
        with pytest.raises(generator.CaptionGenerationError, match=message) as error:
            loader()
        assert "private" not in str(error.value)


def mock_resources():
    processor = Mock()
    inputs = Mock()
    pixels = torch.zeros((1, 3, 224, 224))
    inputs.to.return_value = {"pixel_values": pixels}
    processor.return_value = inputs
    processor.batch_decode.return_value = [" a cat on a couch "]
    model = Mock(device=torch.device("cpu"))
    model.generate.return_value = torch.tensor([[101, 103, 102]])
    return model, processor, inputs


def test_generation_configuration_decoding_and_inference_mode():
    model, processor, inputs = mock_resources()
    progress = []
    model.generate.side_effect = lambda **kwargs: (
        torch.tensor([[101, 103, 102]]) if torch.is_inference_mode_enabled()
        else pytest.fail("Inference must disable autograd")
    )
    image = Image.new("RGB", (30, 40))
    with patch.object(generator, "load_processor", return_value=processor), \
         patch.object(generator, "load_caption_model", return_value=model):
        assert generator.generate_caption(image, progress.append) == "A cat on a couch."
    processor.assert_called_once_with(images=image, return_tensors="pt")
    inputs.to.assert_called_once_with(model.device)
    kwargs = model.generate.call_args.kwargs
    assert kwargs["num_beams"] == generator.NUM_BEAMS > 1
    assert kwargs["max_new_tokens"] == generator.MAX_NEW_TOKENS
    assert kwargs["no_repeat_ngram_size"] == generator.NO_REPEAT_NGRAM_SIZE
    assert kwargs["early_stopping"] is True
    assert kwargs["do_sample"] is False
    processor.batch_decode.assert_called_once()
    assert processor.batch_decode.call_args.kwargs == {"skip_special_tokens": True}
    assert len(progress) == 3
    assert "Loading" in progress[0]
    assert "Analyzing" in progress[1]
    assert "Generating" in progress[2]


@pytest.mark.parametrize("failure,message", [(RuntimeError("bad inference"), "couldn't generate"),
                                           (MemoryError(), "memory"),
                                           (RuntimeError("DefaultCPUAllocator: not enough memory"), "memory")])
def test_inference_errors(failure, message):
    model, processor, _ = mock_resources()
    model.generate.side_effect = failure
    with patch.object(generator, "load_processor", return_value=processor), \
         patch.object(generator, "load_caption_model", return_value=model), \
         pytest.raises(generator.CaptionGenerationError, match=message):
        generator.generate_caption(Image.new("RGB", (10, 10)))


def test_empty_decoded_caption():
    model, processor, _ = mock_resources()
    processor.batch_decode.return_value = [" "]
    with patch.object(generator, "load_processor", return_value=processor), \
         patch.object(generator, "load_caption_model", return_value=model), \
         pytest.raises(generator.CaptionGenerationError, match="another photo"):
        generator.generate_caption(Image.new("RGB", (10, 10)))


def test_no_tensorflow_in_v2_process():
    assert not any(name == "tensorflow" or name.startswith("tensorflow.") for name in sys.modules)
