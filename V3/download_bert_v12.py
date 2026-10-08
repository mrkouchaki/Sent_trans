r"""Download BERT through JFrog, then check that it loads and runs locally.

Put this file in your ewoc_ticket_type project folder. Set HF_ENDPOINT below.
Run in PowerShell:
    .\.venv\Scripts\python.exe .\download_bert_v12.py

No PowerShell environment-variable setup is needed. This script does not train
or change your EWOC classifier. It checks the pretrained BERT encoder on CPU.
"""

import os
from getpass import getpass
from pathlib import Path
from urllib.parse import urlsplit


# ----------------------- SETTINGS: EDIT HERE -----------------------
MODEL_ID = "google-bert/bert-base-uncased"
LOCAL_DIR = Path("models/bert-base-uncased")  # Relative to your terminal folder.

# Paste the EXACT Hugging Face model endpoint supplied by your team.
# Expected structure:
# https://<jfrog-host>/artifactory/api/huggingfaceml/<model-repository>
# Your existing PyPI package index URL is not the model endpoint.
HF_ENDPOINT = "PASTE_YOUR_JFROG_MODEL_ENDPOINT_HERE"

# Leave empty to enter your own JFrog identity token privately when prompted.
# If you choose to put a token here, keep this file private.
HF_TOKEN = ""

ETAG_TIMEOUT_SECONDS = 60
DOWNLOAD_TIMEOUT_SECONDS = 60
# ------------------------------------------------------------------


def main() -> int:
    endpoint = HF_ENDPOINT.strip().rstrip("/")
    if not endpoint or "PASTE_YOUR_" in endpoint:
        print("Set HF_ENDPOINT in this script to your team's JFrog model URL.")
        print("The screenshots did not include that URL. No download attempted.")
        return 2

    parts = urlsplit(endpoint)
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or not any(
            marker in parts.path
            for marker in ("/api/huggingfaceml/", "/api/huggingface/")
        )
        or not parts.path.rstrip("/").rsplit("/", 1)[-1]
        or parts.path.rstrip("/").rsplit("/", 1)[-1]
        in {"huggingfaceml", "huggingface"}
    ):
        print("HF_ENDPOINT must be the HTTPS JFrog Hugging Face model API URL.")
        print("Use the full endpoint from your team's configuration or Set Me Up.")
        return 2

    # Set these BEFORE importing Hugging Face libraries.
    os.environ["HF_ENDPOINT"] = endpoint
    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(ETAG_TIMEOUT_SECONDS)
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(DOWNLOAD_TIMEOUT_SECONDS)
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    try:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import BertModel, BertTokenizer
    except ImportError as exc:
        print(f"Dependency import failed: {exc}")
        print("Use the same virtual environment where your BERT import succeeded.")
        return 1

    token = HF_TOKEN.strip() or getpass("JFrog identity token (hidden): ").strip()
    if not token:
        print("No token entered. No download attempted.")
        return 2

    destination = LOCAL_DIR.expanduser().resolve()
    print(f"Model: {MODEL_ID}", flush=True)
    print(f"JFrog endpoint: {endpoint}", flush=True)
    print(f"Save folder: {destination}", flush=True)
    print("Downloading BERT weights and tokenizer files...", flush=True)

    stage = "Download"
    try:
        folder = snapshot_download(
            repo_id=MODEL_ID,
            revision="main",
            local_dir=str(destination),
            endpoint=endpoint,
            token=token,
            local_files_only=False,
            etag_timeout=ETAG_TIMEOUT_SECONDS,
            max_workers=2,
            # Download one weights format, plus the tokenizer and configuration.
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.txt",
                "README.md",
                "LICENSE",
            ],
        )

        stage = "Local BERT check"
        print("Download complete. Checking BERT locally on CPU...", flush=True)
        tokenizer = BertTokenizer.from_pretrained(folder, local_files_only=True)
        model = BertModel.from_pretrained(
            folder, local_files_only=True, use_safetensors=True
        ).to("cpu")
        model.eval()
        inputs = tokenizer("Please update the E911 database.", return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs).last_hidden_state
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("BERT returned non-finite values during the check.")
    except Exception as exc:
        # Never echo the token if an underlying error message happens to include it.
        detail = str(exc).replace(token, "[REDACTED]")
        print(f"{stage} failed ({type(exc).__name__}): {detail}")
        if stage == "Download":
            print("Check the JFrog model endpoint, token access, and connectivity.")
        else:
            print("Files were downloaded, but the local load/run check did not pass.")
        return 1

    print("SUCCESS: BERT tokenizer, pretrained weights, and CPU forward pass OK.")
    print(f"Local model folder: {destination}")
    print("Use this folder in from_pretrained(..., local_files_only=True).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCancelled.")
        raise SystemExit(130)
