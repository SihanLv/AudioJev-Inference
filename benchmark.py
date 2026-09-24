"""Compare multi-question audio reuse with separate calls on the same model."""

import argparse
import json
import statistics
import time

from audiojev_inference import AudioJev, Noul


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--questions", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.questions < 2 or args.repeats < 1:
        parser.error("need at least two questions and one repeat")
    model = AudioJev(args.model_dir, adapter=args.adapter, device=args.device)
    questions = {str(i): Noul(f"Can speech be heard in this audio? Check {i}.")
                 for i in range(args.questions)}
    model.system_one(state=args.audio, questions=questions)
    grouped, separate = [], []
    group_calls, separate_calls = [], []
    for _ in range(args.repeats):
        start = time.perf_counter()
        response = model.system_one(state=args.audio, questions=questions)
        grouped.append(1000 * (time.perf_counter() - start))
        group_calls.append(response["usage"]["audio_encoder_calls"])
        start = time.perf_counter()
        calls = 0
        for key, question in questions.items():
            response = model.system_one(state=args.audio, questions={key: question})
            calls += response["usage"]["audio_encoder_calls"]
        separate.append(1000 * (time.perf_counter() - start))
        separate_calls.append(calls)
    print(json.dumps({"questions": args.questions, "repeats": args.repeats,
                      "grouped_median_ms": statistics.median(grouped),
                      "separate_median_ms": statistics.median(separate),
                      "grouped_encoder_calls": group_calls,
                      "separate_encoder_calls": separate_calls}, indent=2))


if __name__ == "__main__":
    main()
