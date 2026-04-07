#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from urllib import request


def call_text_embedding(api_url: str, text: str, timeout_sec: int) -> dict:
    payload = json.dumps({"text": text}).encode("utf-8")
    req = request.Request(
        url=api_url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=timeout_sec) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Get BGE text embedding from proxy."
    )
    parser.add_argument("text", help="Input text to embed")
    parser.add_argument(
        "--api-url",
        default="http://localhost:4023/text_embedding",
        help="text_embedding API URL",
    )
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Print the full embedding vector (default: first 16 values only)",
    )
    args = parser.parse_args()

    text = args.text.strip()
    result = call_text_embedding(args.api_url, text, args.timeout_sec)
    embeddings = result["embeddings"]
    shape = result["embeddings_shape"]

    print(f"embeddings_shape={shape}")
    vector = embeddings[0]
    if args.show_all:
        print(vector)
    else:
        print("embedding[:16]=", vector[:16])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
