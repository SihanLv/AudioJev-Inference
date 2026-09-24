"""Optional localhost HTTP service for AudioJev."""

import argparse
import base64
import binascii
import os

from .api import AudioInput, AudioJev


def create_app(model_dir, *, adapter=None, device="cuda:0", merge_adapter=False,
               max_audio_bytes=20_000_000):
    from fastapi import FastAPI, HTTPException
    model = AudioJev(model_dir, adapter=adapter, device=device, merge_adapter=merge_adapter)
    app = FastAPI(title="AudioJev local inference")

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
    parser.add_argument("--model-dir", default=os.environ.get("AUDIOJEV_MODEL_DIR"), required=False)
    parser.add_argument("--adapter", default=os.environ.get("AUDIOJEV_ADAPTER"))
    parser.add_argument("--device", default=os.environ.get("AUDIOJEV_DEVICE", "cuda:0"))
    parser.add_argument("--merge-adapter", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not args.model_dir:
        parser.error("--model-dir or AUDIOJEV_MODEL_DIR is required")
    import uvicorn
    app = create_app(args.model_dir, adapter=args.adapter, device=args.device,
                     merge_adapter=args.merge_adapter)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
