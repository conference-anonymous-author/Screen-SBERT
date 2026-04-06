#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import mimetypes
import uuid
from pathlib import Path
from urllib import request

import numpy as np


def build_multipart_form_data(field_name: str, filename: str, file_bytes: bytes) -> tuple[bytes, str]:
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    lines = [
        f"--{boundary}".encode("utf-8"),
        (
            f'Content-Disposition: form-data; name="{field_name}"; filename="{Path(filename).name}"'
        ).encode("utf-8"),
        f"Content-Type: {content_type}".encode("utf-8"),
        b"",
        file_bytes,
        f"--{boundary}--".encode("utf-8"),
        b"",
    ]
    body = b"\r\n".join(lines)
    return body, f"multipart/form-data; boundary={boundary}"


def call_screen_sbert_embed_npy(api_url: str, image_path: Path, timeout_sec: int) -> bytes:
    image_bytes = image_path.read_bytes()
    body, content_type = build_multipart_form_data("file", image_path.name, image_bytes)

    req = request.Request(
        url=api_url,
        data=body,
        method="POST",
        headers={"Content-Type": content_type},
    )
    with request.urlopen(req, timeout=timeout_sec) as resp:
        return resp.read()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Call /screen_sbert_embed end-to-end and save one functional semantics embedding vector."
    )
    parser.add_argument("image_path", help="Input image path")
    parser.add_argument(
        "--api-url",
        default="http://localhost:4023/screen_sbert_embed?return_type=npy",
        help="screen_sbert_embed API URL",
    )
    parser.add_argument(
        "--output-dir",
        default="example/results",
        help="Output directory",
    )
    parser.add_argument("--timeout-sec", type=int, default=120)
    args = parser.parse_args()

    image_path = Path(args.image_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_bytes = call_screen_sbert_embed_npy(args.api_url, image_path, args.timeout_sec)
    embedding = np.load(io.BytesIO(npy_bytes), allow_pickle=False).astype(np.float32, copy=False)

    output_path = output_dir / f"{image_path.stem}_screen_sbert_embedding.npy"
    np.save(output_path, embedding)

    print(output_path)
    print(f"shape={tuple(embedding.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

