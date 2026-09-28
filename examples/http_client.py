"""Send an audio question to an AudioJev-Inference HTTP server."""

import argparse
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--options", nargs="+", required=True)
    args = parser.parse_args()
    if not 2 <= len(args.options) <= 36:
        parser.error("provide between 2 and 36 candidate descriptions")
    payload = {
        "state": {"audio_base64": base64.b64encode(Path(args.audio).read_bytes()).decode("ascii")},
        "questions": {"answer": {
            "type": "choice", "instructions": args.question,
            "criteria": {str(i): text for i, text in enumerate(args.options)},
        }},
    }
    request = Request(args.url.rstrip("/") + "/v1/systemone",
                      data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        print(json.dumps(json.load(response), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
