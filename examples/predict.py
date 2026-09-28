"""Ask a multiple-choice question about an audio file using AudioJev-Inference."""

import argparse
import json

from audiojev_inference import DEFAULT_MODEL, AudioJev, Choice


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--options", nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 2 <= len(args.options) <= 36:
        parser.error("provide between 2 and 36 candidate descriptions")
    model = AudioJev(args.model, device=args.device, revision=args.revision,
                     local_files_only=args.local_files_only)
    result = model.system_one(
        state=args.audio,
        questions={"answer": Choice(args.question, {str(i): text for i, text in enumerate(args.options)})},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
