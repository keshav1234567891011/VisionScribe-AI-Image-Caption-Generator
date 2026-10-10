from io import BytesIO
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest

import caption_generator as generator


def upload(color="red"):
    file = BytesIO()
    file.name = "photo.png"
    Image.new("RGB", (20, 30), color).save(file, format="PNG")
    return file


def test_initial_page_is_lazy_and_disabled():
    with patch.object(generator, "load_caption_model") as loader:
        app = AppTest.from_file("main.py").run(timeout=30)
        assert not app.exception
        assert app.title[0].value == "VisionScribe"
        assert app.button[0].disabled
        loader.assert_not_called()


def test_generate_rerun_regenerate_and_clear():
    with patch("streamlit.file_uploader", return_value=upload()) as uploader, \
         patch.object(generator, "generate_caption", return_value="A red image.") as generate:
        app = AppTest.from_file("main.py").run(timeout=30)
        assert not app.button[0].disabled
        assert app.button[0].label == "Generate Caption"
        app.button[0].click().run()
        assert not app.exception
        assert app.session_state["caption"] == "A red image."
        assert app.button[0].label == "Regenerate"
        app.run()
        generate.assert_called_once()
        app.button[0].click().run()
        assert generate.call_count == 2
        app.button[1].click().run()
        assert app.session_state["upload_number"] == 1
        assert "caption" not in app.session_state
        uploader.return_value = None
        app.run()
        assert app.button[0].disabled


def test_new_image_clears_previous_caption():
    with patch("streamlit.file_uploader", return_value=upload()) as uploader, \
         patch.object(generator, "generate_caption", return_value="A red image.") as generate:
        app = AppTest.from_file("main.py").run(timeout=30)
        app.button[0].click().run()
        uploader.return_value = upload("blue")
        app.run()
        assert "caption" not in app.session_state
        assert "inference_seconds" not in app.session_state
        generate.assert_called_once()


def test_bad_upload_and_prediction_errors_are_friendly():
    with patch("streamlit.file_uploader", return_value=BytesIO(b"bad")):
        app = AppTest.from_file("main.py").run(timeout=30)
        assert not app.exception
        assert app.button[0].disabled
        assert "couldn't open" in app.error[0].value
    with patch("streamlit.file_uploader", return_value=upload()), \
         patch.object(generator, "generate_caption",
                      side_effect=generator.CaptionGenerationError("Please try again.")):
        app = AppTest.from_file("main.py").run(timeout=30)
        app.button[0].click().run()
        assert not app.exception
        assert app.error[0].value == "Please try again."
