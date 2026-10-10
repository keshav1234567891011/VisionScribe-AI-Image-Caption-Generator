"""Make unit tests fail if an unmocked loader tries to access Hugging Face."""

import os

if os.environ.get("RUN_MODEL_INTEGRATION") != "1":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
