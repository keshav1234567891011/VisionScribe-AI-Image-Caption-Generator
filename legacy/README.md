# Legacy VisionScribe v1

This is the preserved educational **VGG16 + LSTM + Flickr8k** implementation.
The production entry point `main.py` at the repository root runs v2 and never
imports this package. Use v1 only in a separate environment/process.

## Running the reference app

From the repository root, create a separate Python 3.11 virtual environment,
activate it, then run:

```bash
python -m pip install -r requirements-legacy.txt
python -m streamlit run legacy/main.py
```

Do not install this dependency file in the v2 or Streamlit Cloud environment.
The first v1 request downloads VGG16's ImageNet weights if not cached.

The original `best_model.h5`, `tokenizer.pkl`, and notebook have been moved here
without changing their contents. The archived loader resolves their paths
relative to `legacy/caption_generator.py`. Its Python modules use explicit
`legacy.*` imports so they cannot accidentally select the v2 backend.

## Confirmed architecture

```text
RGB image → 224 × 224 nearest-neighbor resize → VGG16 preprocessing
→ ImageNet-pretrained VGG16 fc2 → 4096-dimensional visual features
→ merged image branch and LSTM text branch → greedy word predictions
```

The image branch uses Dropout(0.4) and Dense(256, ReLU). The text branch uses a
35-index sequence, Embedding(8485 × 256, mask_zero=True), Dropout(0.4), and
LSTM(256). Their representations are added, followed by Dense(256, ReLU) and
Dense(8485, softmax). Inputs are ordered `[image_features, sequence]`.

The supplied tokenizer has 8,484 indexed words. Generation begins with
`startseq` (index 1), pre-pads/pre-truncates to 35 indices, chooses `argmax`, and
stops at `endseq` (index 2), an unmapped index, a repeated start token, or 35 steps.
Display formatting removes these internal markers. Resources use Streamlit
caching, and the UI generates only on a button click.

## Training provenance and historical evaluation

The notebook contains a Flickr8k training workflow and saved outputs. It extracts
VGG16 features separately; it does not train VGG16 from scratch or fine-tune it.
The model's H5 metadata reports Keras 2.6.0 serialization; the archived loader
uses TensorFlow 2.15.1/Keras 2 and `compile=False` for inference compatibility.

The initial project upload introduced the model, tokenizer, and notebook together.
Available history does not establish the original trainer or an external source
repository/license. The artifacts are supplied and reused, not retrained here.

Saved notebook results are **BLEU-1 = 0.533212** and **BLEU-2 = 0.305637** using
NLTK corpus BLEU. They are historical **v1** outputs, not reproduced or applicable
to v2. The notebook uses the first 90% of image IDs in caption-file insertion
order for training and the last 10% for evaluation (810 images in saved output).
It fits the tokenizer before splitting and scores with sequence markers retained.
The original images, complete caption file, feature cache, exact split manifest,
and historical evaluation environment are absent, preventing reproduction from
this checkout alone. No v2 BLEU/CIDEr score is claimed.

## Reference tests

With the legacy dependencies and pytest installed in that separate environment:

```bash
python -m pytest legacy/tests -v
```

These preserved tests cover original preprocessing, decoder compatibility,
sequence behavior, caching, and the reference Streamlit workflow. The default
v2 test suite excludes them. The notebook requires additional research packages
and the Flickr8k dataset to execute; running the app never executes or retrains it.

## Educational value

V1 demonstrates CNN feature extraction, transfer learning, tokenization, recurrent
language generation, and greedy decoding. V2 extends the project with transformer
vision-language inference and modern deployment/resource management. See the
[current README](../README.md) and [original UI capture](docs/images/visionscribe.png).
