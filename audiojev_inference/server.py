"""AudioJev-Inference HTTP service."""

import argparse
import base64
import binascii
import os

from .api import DEFAULT_MODEL, AudioInput, AudioJev


def create_app(model_dir=DEFAULT_MODEL, *, adapter=None, device="cuda:0", merge_adapter=False,
               revision=None, local_files_only=False, max_audio_bytes=20_000_000):
    from fastapi import FastAPI, HTTPException
    model = AudioJev(model_dir, adapter=adapter, device=device, merge_adapter=merge_adapter,
                    revision=revision, local_files_only=local_files_only)
    app = FastAPI(title="AudioJev-Inference", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model.model}

    @app.post("/v1/systemone")
    def system_one(request: dict):
        try:
            if request.get("model", model.model) != model.model:
                raise ValueError(f"model must be {model.model!r}")
            if not isinstance(request.get("questions"), dict):
                raise ValueError("questions must be a mapping")
            state = request.get("state")
            if not isinstance(state, dict):
                raise ValueError("state must be an object")
            if set(state) != {"audio_base64"}:
                raise ValueError("state must contain only audio_base64")
            encoded = state["audio_base64"]
            if not isinstance(encoded, str) or len(encoded) > 4 * max_audio_bytes // 3 + 8:
                raise ValueError("audio_base64 exceeds limit")
            data = base64.b64decode(encoded, validate=True)
            if len(data) > max_audio_bytes:
                raise ValueError("decoded audio exceeds limit")
            return model.system_one(state=AudioInput(data=data), questions=request["questions"])
        except (ValueError, binascii.Error) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", "--model-dir", dest="model_dir",
                        default=os.environ.get("AUDIOJEV_MODEL") or os.environ.get("AUDIOJEV_MODEL_DIR") or DEFAULT_MODEL,
                        help="Hugging Face model ID or local model directory (default: %(default)s)")
    parser.add_argument("--revision", help="Hugging Face branch, tag, or commit SHA")
    parser.add_argument("--local-files-only", action="store_true", help="use local files or an already cached Hub snapshot")
    parser.add_argument("--adapter", default=os.environ.get("AUDIOJEV_ADAPTER"))
    parser.add_argument("--device", default=os.environ.get("AUDIOJEV_DEVICE", "cuda:0"))
    parser.add_argument("--merge-adapter", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn
    app = create_app(args.model_dir, adapter=args.adapter, device=args.device,
                     merge_adapter=args.merge_adapter, revision=args.revision,
                     local_files_only=args.local_files_only)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
