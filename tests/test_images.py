from io import BytesIO
from unittest.mock import patch

from PIL import Image
import pytest

from utils import format_caption, open_uploaded_image


def image_upload(mode="RGB", file_format="PNG"):
    upload = BytesIO()
    Image.new(mode, (12, 20)).save(upload, format=file_format)
    upload.seek(0)
    return upload


@pytest.mark.parametrize("mode,file_format", [("RGB", "JPEG"), ("RGBA", "PNG"),
                                              ("L", "PNG"), ("RGBA", "WEBP")])
def test_supported_images_are_rgb(mode, file_format):
    image = open_uploaded_image(image_upload(mode, file_format))
    assert image.mode == "RGB"
    assert image.size == (12, 20)


def test_exif_orientation_is_corrected():
    image = Image.new("RGB", (12, 20))
    exif = image.getexif()
    exif[274] = 6
    upload = BytesIO()
    image.save(upload, format="JPEG", exif=exif)
    result = open_uploaded_image(upload)
    assert result.size == (20, 12)
    assert 274 not in result.getexif()


@pytest.mark.parametrize("content", [b"broken", b"", image_upload(file_format="JPEG").getvalue()[:-100]],
                         ids=["invalid", "empty", "truncated-jpeg"])
def test_corrupt_images_have_friendly_errors(content):
    with pytest.raises(ValueError, match="couldn't open"):
        open_uploaded_image(BytesIO(content))


def test_renamed_unsupported_content_is_rejected():
    upload = image_upload(file_format="BMP")
    upload.name = "pretend.png"
    with pytest.raises(ValueError, match="valid JPG"):
        open_uploaded_image(upload)


def test_byte_and_pixel_limits():
    with patch("utils.MAX_UPLOAD_BYTES", 1), pytest.raises(ValueError, match="too large"):
        open_uploaded_image(image_upload())
    with patch("utils.MAX_IMAGE_PIXELS", 1), pytest.raises(ValueError, match="16 million"):
        open_uploaded_image(image_upload())


@pytest.mark.parametrize("raw,expected", [("  a cat\n on a sofa  ", "A cat on a sofa."),
                                         ("NASA rocket!", "NASA rocket!"), (" ", "")])
def test_caption_cleanup_preserves_content(raw, expected):
    assert format_caption(raw) == expected
