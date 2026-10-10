"""VisionScribe v2 Streamlit UI; inference lives in caption_generator.py."""

from hashlib import sha256
from io import BytesIO
import logging
from time import perf_counter

import streamlit as st

from caption_generator import CaptionGenerationError, MODEL_ID, generate_caption
from utils import open_uploaded_image

LOGGER = logging.getLogger(__name__)


def clear_result() -> None:
    """Remove only the result associated with this session's current image."""
    st.session_state.pop("caption", None)
    st.session_state.pop("inference_seconds", None)


def clear_upload() -> None:
    """Reset the upload widget without clearing shared model resources."""
    st.session_state.upload_number += 1
    st.session_state.pop("image_id", None)
    clear_result()


def main() -> None:
    """Render a two-column preview and caption workflow with lazy inference."""
    st.set_page_config(page_title="VisionScribe – AI Image Caption Generator",
                       page_icon="✍️", layout="wide")
    st.markdown("""<style>
        .block-container {max-width: 1360px; padding-top: 2.8rem;}
        h1 {letter-spacing: -0.045em; font-size: 3.4rem;}
        h3 {letter-spacing: -0.02em;}
        .hero-label {color: #6366f1; font-weight: 650; letter-spacing: .12em;
                     font-size: .8rem; margin-bottom: .6rem;}
        .model-badges {color: #657084; font-size: .88rem; margin: 1rem 0 1.8rem;}
        div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 18px;}
        </style>""", unsafe_allow_html=True)

    with st.sidebar:
        st.header("About VisionScribe v2")
        st.markdown("**Model**\n\nGIT-base-COCO\n\n"
                    "**Architecture**\n\nVision encoder + Transformer decoder\n\n"
                    "**Task**\n\nImage Caption Generation\n\n"
                    "**Framework**\n\nPyTorch + Hugging Face Transformers")
        st.divider()
        st.caption("Image → Vision encoder / image tokens → Transformer decoder "
                   "→ Natural-language caption")
        st.link_button("View model card", "https://huggingface.co/microsoft/git-base-coco")
        with st.expander("Legacy VisionScribe v1"):
            st.write("The original VGG16 + LSTM + Flickr8k implementation is "
                     "preserved in legacy/ for learning and reference. V2 uses GIT exclusively.")
        st.caption("This project uses Microsoft's pretrained, COCO-fine-tuned model.")

    st.markdown('<div class="hero-label">VISION TO LANGUAGE · V2</div>', unsafe_allow_html=True)
    st.title("VisionScribe")
    st.subheader("AI-Powered Image Caption Generator")
    st.write("Turn images into natural-language descriptions with a modern vision-language model.")
    st.markdown('<div class="model-badges">GIT Transformer &nbsp; · &nbsp; COCO Captioning '
                '&nbsp; · &nbsp; Hugging Face Transformers</div>', unsafe_allow_html=True)

    st.session_state.setdefault("upload_number", 0)
    with st.container(border=True):
        uploaded = st.file_uploader(
            "Upload a JPG, JPEG, PNG, or WEBP image", type=["jpg", "jpeg", "png", "webp"],
            key=f"image_upload_{st.session_state.upload_number}",
            help="Maximum 10 MB and 16 million pixels. Photos of everyday scenes work best.",
        )
    image = None
    if uploaded is not None:
        content = uploaded.getvalue()
        image_id = sha256(content).hexdigest()
        if st.session_state.get("image_id") != image_id:
            st.session_state.image_id = image_id
            clear_result()
        try:
            image = open_uploaded_image(BytesIO(content))
        except ValueError as error:
            st.error(str(error))
    else:
        st.session_state.pop("image_id", None)
        clear_result()

    preview_column, result_column = st.columns([1.3, 1], gap="large")
    with preview_column:
        with st.container(border=True):
            if image is not None:
                st.image(image, use_container_width=True)
            else:
                st.subheader("A picture, in words")
                st.write("Choose a photo to see a preview here. Generate a concise "
                         "description of its visible scene with one click.")
                st.caption("People · Animals · Places · Everyday objects")
    with result_column:
        if image is not None:
            st.text(uploaded.name)
            st.caption(f"{image.width:,} × {image.height:,} pixels · {len(content) / 1024:.0f} KB")
        has_caption = bool(st.session_state.get("caption"))
        generate = st.button("Regenerate" if has_caption else "Generate Caption",
                             type="primary", disabled=image is None, use_container_width=True)
        st.button("Clear Image", on_click=clear_upload, disabled=uploaded is None,
                  use_container_width=True)

        if generate and image is not None:
            clear_result()
            started = perf_counter()
            with st.status("Preparing VisionScribe...", expanded=True) as status:
                try:
                    caption = generate_caption(
                        image, on_progress=lambda message: status.update(label=message),
                    )
                except CaptionGenerationError as error:
                    status.update(label="Caption unavailable", state="error", expanded=False)
                    st.error(str(error))
                except Exception:
                    LOGGER.exception("Unexpected UI generation failure")
                    status.update(label="Caption unavailable", state="error", expanded=False)
                    st.error("Something went wrong. Please try again shortly.")
                else:
                    st.session_state.caption = caption
                    st.session_state.inference_seconds = perf_counter() - started
                    status.update(label="Caption ready", state="complete", expanded=False)
                    # Render the Regenerate label and download control consistently.
                    st.rerun()

        if st.session_state.get("caption"):
            with st.container(border=True):
                st.subheader("Generated Caption")
                st.write(st.session_state.caption)
            st.caption(f"Model: {MODEL_ID} · Generation: Beam Search")
            st.caption(f"Inference time: {st.session_state.inference_seconds:.1f} s "
                       "(includes loading on the first request)")
            st.download_button("Download caption", st.session_state.caption,
                               file_name="caption.txt", mime="text/plain",
                               use_container_width=True)
            st.caption("Regenerate uses the same deterministic settings. "
                       "AI captions may miss details or describe them incorrectly.")
        elif image is not None:
            st.caption("Ready to generate. The first request loads the model; "
                       "later requests reuse it.")


if __name__ == "__main__":
    main()
