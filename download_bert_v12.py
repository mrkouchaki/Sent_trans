r"""Download BERT through the corporate proxy, then check it locally.

Save in your ewoc_ticket_type project folder and run in PowerShell:
    .\.venv\Scripts\python.exe .\download_bert_v12.py

All settings are below. Uses Windows trusted certificates and keeps HTTPS
certificate verification enabled. No JFrog token or separate shell setup needed.
This downloads the pretrained encoder; it does not train your classifier.
"""

import os
from pathlib import Path
import ssl
from tempfile import TemporaryDirectory


# --------------------------- SETTINGS -----------------------------
PROXY_URL = "http://cso.proxy.att.com:8888"
HF_ENDPOINT = "https://huggingface.co"
MODEL_ID = "google-bert/bert-base-uncased"
LOCAL_DIR = Path("models/bert-base-uncased")  # Relative to your terminal folder.

# Usually leave empty: Windows trusted certificates are used automatically.
# If IT gives you an additional CA certificate bundle, put its PEM path here.
CA_BUNDLE = ""  # Example: r"C:\certificates\corporate-ca.pem"

ETAG_TIMEOUT_SECONDS = 60
DOWNLOAD_TIMEOUT_SECONDS = 120
# ------------------------------------------------------------------


def configure_download() -> None:
    """Configure this Python process before importing Hugging Face libraries."""
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ[name] = PROXY_URL

    # Ensure old proxy bypass settings do not bypass this proxy for Hugging Face.
    for name in ("NO_PROXY", "no_proxy"):
        os.environ[name] = "localhost,127.0.0.1,::1"

    os.environ["HF_ENDPOINT"] = HF_ENDPOINT
    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(ETAG_TIMEOUT_SECONDS)
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(DOWNLOAD_TIMEOUT_SECONDS)
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    # Use ordinary HTTP downloads through the proxy.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    # This public model needs no token. Do not send a previous JFrog credential.
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        os.environ.pop(name, None)


def configure_certificates(bundle_path: Path) -> int:
    """Make OS-trusted CAs available to both requests and httpx clients."""
    # On Windows this loads the Windows CA and ROOT certificate stores.
    context = ssl.create_default_context()

    try:
        import certifi
    except ImportError:
        pass
    else:
        context.load_verify_locations(cafile=certifi.where())

    # Preserve any extra CA bundles already provided by the organization.
    extra_bundles = {
        value.strip()
        for value in (
            CA_BUNDLE,
            os.environ.get("REQUESTS_CA_BUNDLE", ""),
            os.environ.get("CURL_CA_BUNDLE", ""),
        )
        if value.strip()
    }
    for filename in extra_bundles:
        context.load_verify_locations(
            cafile=str(Path(filename).expanduser().resolve())
        )

    certificates = context.get_ca_certs(binary_form=True)
    if not certificates:
        raise RuntimeError("No trusted CA certificates were available.")

    bundle_path.write_text(
        "".join(ssl.DER_cert_to_PEM_cert(cert) for cert in certificates),
        encoding="ascii",
    )
    # Older huggingface_hub uses requests; newer versions use httpx.
    os.environ["REQUESTS_CA_BUNDLE"] = str(bundle_path)
    os.environ["SSL_CERT_FILE"] = str(bundle_path)
    return len(certificates)


def main() -> int:
    configure_download()
    destination = LOCAL_DIR.expanduser().resolve()

    print(f"Model: {MODEL_ID}", flush=True)
    print(f"Proxy: {PROXY_URL}", flush=True)
    print(f"Source: {HF_ENDPOINT}", flush=True)
    print(f"Save folder: {destination}", flush=True)

    stage = "Certificate setup"
    try:
        # Keep the CA file available until all downloads and checks finish.
        with TemporaryDirectory(prefix="bert-proxy-ca-") as cert_dir:
            count = configure_certificates(Path(cert_dir) / "trusted-ca.pem")
            print(f"HTTPS verification enabled ({count} trusted CAs).", flush=True)

            stage = "Dependency import"
            import torch
            from huggingface_hub import snapshot_download
            from transformers import BertModel, BertTokenizer

            print("BERT import OK. Downloading model files through the proxy...", flush=True)
            stage = "Download"
            folder = snapshot_download(
                repo_id=MODEL_ID,
                revision="main",
                local_dir=str(destination),
                endpoint=HF_ENDPOINT,
                token=False,
                local_files_only=False,
                etag_timeout=ETAG_TIMEOUT_SECONDS,
                max_workers=2,
                # One weights format, plus configuration and tokenizer files.
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
            tokenizer = BertTokenizer.from_pretrained(
                folder, local_files_only=True, token=False
            )
            model = BertModel.from_pretrained(
                folder, local_files_only=True, use_safetensors=True, token=False
            ).to("cpu")
            model.eval()
            inputs = tokenizer("Please update the E911 database.", return_tensors="pt")
            with torch.no_grad():
                output = model(**inputs).last_hidden_state
            if not bool(torch.isfinite(output).all()):
                raise RuntimeError("BERT returned non-finite values during the check.")

    except Exception as exc:
        print(f"\n{stage} failed ({type(exc).__name__}): {exc}", flush=True)
        if stage == "Certificate setup":
            print("Check any configured CA file paths. IT can supply a PEM CA bundle.")
        elif stage == "Dependency import":
            print("Use the same virtual environment where your BERT import succeeded.")
        elif stage == "Download":
            print("Check connectivity to the corporate proxy.")
            print("407 means proxy authentication is required; ask IT for its setup.")
            print("For certificate errors, ask IT for the CA bundle and set CA_BUNDLE.")
            print("For access-denied errors, ask IT to check the blocked download URL.")
        else:
            print("Model files downloaded, but the local load/run check did not pass.")
        return 1

    print("\nSUCCESS: BERT tokenizer, pretrained weights, and CPU forward pass OK.")
    print(f"Local model folder: {destination}")
    print("Use this folder in from_pretrained(..., local_files_only=True).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCancelled.")
        raise SystemExit(130)
