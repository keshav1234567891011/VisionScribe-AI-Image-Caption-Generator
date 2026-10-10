"""Upload validation and display formatting, independent of the model backend."""

from typing import BinaryIO

from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000


def open_uploaded_image(upload: BinaryIO) -> Image.Image:
    """Validate size/content, correct EXIF orientation, and fully decode to RGB."""
    upload.seek(0, 2)
    upload_size = upload.tell()
    upload.seek(0)
    if upload_size > MAX_UPLOAD_BYTES:
        raise ValueError("This file is too large. Please choose an image under 10 MB.")
    try:
        with Image.open(upload) as image:
            if image.format not in SUPPORTED_IMAGE_FORMATS:
                raise ValueError("Unsupported image format")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image dimensions exceed the pixel limit")
            return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError,
            Image.DecompressionBombError) as error:
        raise ValueError(
            "We couldn't open this image. Use a valid JPG, JPEG, PNG, or WEBP "
            "with at most 16 million pixels."
        ) from error


def format_caption(caption: str) -> str:
    """Normalize whitespace and sentence casing without inventing caption content."""
    caption = " ".join(caption.split()).strip()
    if not caption:
        return ""
    caption = caption[0].upper() + caption[1:]
    if caption[-1] not in ".!?":
        caption += "."
    return caption
