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


def call_parse_gui_npz(api_url: str, image_path: Path, timeout_sec: int) -> bytes:
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
        description="Call /parse_gui and save four embedding files for Screen-SBERT input."
    )
    parser.add_argument("image_path", help="Input image path")
    parser.add_argument(
        "--api-url",
        default="http://localhost:4023/parse_gui?return_type=npz",
        help="parse_gui API URL",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: example/results/<image_stem>_parse_gui)",
    )
    parser.add_argument("--timeout-sec", type=int, default=120)
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if args.output_dir is None:
        output_dir = Path("example/results") / f"{image_path.stem}_parse_gui"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_bytes = call_parse_gui_npz(args.api_url, image_path, args.timeout_sec)

    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as npz_data:
        bbox = npz_data["bbox"].astype(np.float32, copy=False)
        function_embedding = npz_data["function_embedding"].astype(np.float32, copy=False)
        text_embedding = npz_data["text_embedding"].astype(np.float32, copy=False)
        vision_embedding = npz_data["vision_embedding"].astype(np.float32, copy=False)

    np.save(output_dir / "bbox.npy", bbox)
    np.save(output_dir / "function_embedding.npy", function_embedding)
    np.save(output_dir / "text_embedding.npy", text_embedding)
    np.save(output_dir / "vision_embedding.npy", vision_embedding)

    print(output_dir / "bbox.npy")
    print(output_dir / "function_embedding.npy")
    print(output_dir / "text_embedding.npy")
    print(output_dir / "vision_embedding.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

