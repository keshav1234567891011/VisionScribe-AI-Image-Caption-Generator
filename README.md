# VisionScribe - AI Image Caption Generator

**A computer vision and natural language processing portfolio project that turns photographs into short captions using a VGG16 encoder, an LSTM decoder, and Streamlit.**

[**Try the Live Demo**](https://image-caption-generator-mvzdrcuzwoepbfhnzmsa4b.streamlit.app/) | [Source Code](https://github.com/keshav1234567891011/VisionScribe-AI-Image-Caption-Generator)

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![TensorFlow 2.15.1](https://img.shields.io/badge/TensorFlow-2.15.1-FF6F00?logo=tensorflow&logoColor=white)
![Streamlit 1.41.1](https://img.shields.io/badge/Streamlit-1.41.1-FF4B4B?logo=streamlit&logoColor=white)
![Architecture VGG16 + LSTM](https://img.shields.io/badge/Architecture-VGG16%20%2B%20LSTM-6C63FF)
![Dataset Flickr8k](https://img.shields.io/badge/Dataset-Flickr8k-2E8B57)

## Overview

VisionScribe connects a pretrained visual encoder to a trained language decoder to generate captions one word at a time. The application packages the supplied model and tokenizer into an interactive upload, preview, and inference workflow. Running the app requires no retraining or dataset download.

The portfolio work demonstrates transfer learning, multimodal inference, model artifact integration, modular Python application design, resource caching, and automated regression testing. The base captioning implementation is acknowledged under [Attribution](#attribution).

## Features

- Upload JPG, JPEG, PNG, or WEBP images and preview them immediately.
- Correct EXIF camera orientation and convert uploads to RGB.
- Generate captions on demand with a progress indicator and a dedicated result card.
- Remove internal sequence markers and format predictions as sentences.
- Preserve captions across Streamlit reruns; clear the session for another upload.
- Cache the encoder, decoder, and tokenizer for reuse.
- Display helpful errors for invalid images, missing artifacts, and inference failures.

## VGG16-LSTM Architecture

```text
Image -> RGB -> 224 x 224 -> VGG16 preprocessing -> VGG16 fc2 (4096)
                                                        |
                                             Dropout(0.4) -> Dense(256, ReLU)
                                                        |
                                                        +--> Add -> Dense(256, ReLU)
                                                        |                |
Caption prefix -> Tokenizer -> Pre-pad to 35             |       Dense(vocabulary, softmax)
                         |                              |                |
                Embedding(256, mask_zero=True)          |       Greedy next-word selection
                         |                              |                |
                   Dropout(0.4) -> LSTM(256) ------------+       Append word and repeat
```

The encoder uses ImageNet-pretrained VGG16 with `include_top=True` and takes the output of `fc2`, excluding the final ImageNet classification layer. Each image produces a `(1, 4096)` feature vector. The decoder accepts that vector and a `(1, 35)` token sequence, merges the image and text branches by addition, and predicts a probability distribution over the vocabulary.

The original `best_model.h5` decoder loads with `compile=False`; `tokenizer.pkl` supplies the matching vocabulary. Layer details follow the included training notebook. The app reuses these artifacts rather than fitting a new tokenizer or training a new model.

## Dataset and Training Context

The base model uses **Flickr8k**, a collection of approximately 8,000 photographs with five human-written captions per image. Its everyday scenes support learning associations between visual features and descriptive language.

The included notebook documents:

- Extraction and storage of VGG16 image features.
- Lowercasing captions, filtering single-character words, and adding `startseq` / `endseq` markers.
- Tokenization and padded caption prefixes paired with next-word targets.
- An ordered 90% / 10% split by image ID, rather than the standard predefined dataset partitions.
- Decoder training with Adam and categorical cross-entropy, plus BLEU-1 / BLEU-2 evaluation code.

The notebook is historical training documentation. This README does not claim a newly reproduced benchmark or verified test-set score. The dataset is not needed for app inference; running the full notebook requires the dataset, adjusted paths, and additional research dependencies.

## Image Preprocessing and Caption Workflow

1. **Decode the upload:** Pillow opens the image, corrects EXIF orientation, and converts it to RGB.
2. **Prepare the input:** Resize to `224 x 224` using nearest-neighbor interpolation, convert to `float32`, and add a batch dimension to obtain `(1, 224, 224, 3)`.
3. **Normalize for VGG16:** Apply `tensorflow.keras.applications.vgg16.preprocess_input`, which changes RGB to BGR and subtracts ImageNet channel means.
4. **Encode the image:** Extract the 4096-value `fc2` feature vector.
5. **Decode a caption:** Begin with `startseq`, tokenize and pre-pad the current prefix to 35 positions, then choose the highest-probability next word using `argmax`.
6. **Stop and format:** Stop at `endseq`, an unknown vocabulary index, a repeated `startseq`, or the 35-step limit. Remove sequence markers, capitalize the first letter, and add punctuation when needed.

For illustration, `startseq little girl in pink dress endseq` becomes **Little girl in pink dress.** This illustrates formatting, not a guaranteed prediction. The formatter preserves the model's words without rewriting grammar.

## Technology Stack

| Component | Technology |
| --- | --- |
| Runtime | Python 3.11 |
| Deep learning | TensorFlow 2.15.1 and bundled Keras 2 |
| Visual encoder | ImageNet-pretrained VGG16 |
| Caption decoder | Embedding, LSTM, and Dense layers |
| Web interface | Streamlit 1.41.1 |
| Numerical processing | NumPy (`>=1.23.5,<2`) |
| Image handling | Pillow (`>=10.4,<12`) |
| Tokenizer compatibility | Keras-Preprocessing 1.1.2 |
| Regression checks | Python unittest and Streamlit AppTest |

## Project Structure

```text
VisionScribe-AI-Image-Caption-Generator/
|-- main.py                        # Streamlit interface and session workflow
|-- caption_generator.py           # Cached loaders, preprocessing, greedy decoding
|-- utils.py                       # Upload handling and caption formatting
|-- tests/test_caption_generator.py # Inference and UI regression checks
|-- best_model.h5                  # Trained caption decoder
|-- tokenizer.pkl                  # Matching saved vocabulary
|-- image-caption-generator.ipynb  # Historical training/reference notebook
|-- requirements.txt               # Runtime dependencies
|-- .streamlit/config.toml          # Interface theme
|-- .gitignore
`-- README.md
```

## Installation and Local Usage

Use **Python 3.11** with the pinned TensorFlow version to preserve compatibility with the legacy H5 model.

```bash
git clone https://github.com/keshav1234567891011/VisionScribe-AI-Image-Caption-Generator.git
cd VisionScribe-AI-Image-Caption-Generator
python -m venv .venv
```

Activate the virtual environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies and launch the app:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run main.py
```

Keep `best_model.h5` and `tokenizer.pkl` beside the Python files. Open the local URL printed by Streamlit, upload a photo, and select **Generate Caption**.

The first generation request loads the model resources and downloads ImageNet VGG16 weights if they are absent from the Keras cache. Initial internet access and sufficient memory are required. Resources stay cached across reruns until the server restarts.

## Testing

From the repository root, after installing dependencies, run:

```bash
python -m unittest discover -s tests -v
```

The existing suite checks caption formatting, supported uploads, RGB preprocessing, pre-padding, decoding stop conditions, resource caching, missing-artifact errors, decoder/tokenizer compatibility, and Streamlit generation and reset behavior. Encoder calls are mocked where needed, so the suite does not download VGG16 weights; artifact compatibility checks still load the supplied decoder and tokenizer.

For a manual smoke test, upload a valid photo, generate a caption, confirm it survives a rerun, then clear the upload. Also try an invalid image to check error feedback. These functional checks do not measure caption quality; that requires a held-out dataset and a reproducible evaluation protocol.

## Limitations

- Flickr8k's vocabulary and scene coverage limit generalization to unfamiliar objects, specialized imagery, and complex scenes.
- Captions can omit details, repeat words, contain grammatical errors, or describe objects incorrectly.
- Greedy decoding selects one next word at a time and may miss more fluent sequences.
- Resizing to a square can distort the input; generation is capped at 35 decoding steps.
- VGG16 has substantial memory requirements, and initial loading can be slow on shared hosting.
- The app is an image description demo; it does not reliably read text or provide exhaustive accessibility descriptions.
- No newly verified caption-quality metrics are reported here.

## Potential Extensions

- Compare greedy decoding with beam search under a reproducible evaluation protocol.
- Add batch uploads and downloadable captions.
- Evaluate on a documented held-out split using captioning metrics and qualitative error analysis.
- Explore alternative encoders and language decoders as separate experiments.

## Attribution

VisionScribe builds on an existing open-source VGG16-LSTM image captioning implementation trained on Flickr8k. The exact upstream implementation has not yet been identified; credit for the base training work is not claimed as original to this application. If you know the source, please [open an issue](https://github.com/keshav1234567891011/VisionScribe-AI-Image-Caption-Generator/issues) so attribution can be made precise.

This repository's application work includes modular inference code, a Streamlit interface, image upload and preprocessing, cached resources, error handling, session-state management, automated tests, deployment configuration, and project documentation.

Acknowledgments: the Flickr8k dataset creators and image contributors; the VGG16 researchers Karen Simonyan and Andrew Zisserman; and the TensorFlow/Keras, Streamlit, NumPy, and Pillow communities. Dataset, pretrained weights, and third-party software remain subject to their respective terms.
