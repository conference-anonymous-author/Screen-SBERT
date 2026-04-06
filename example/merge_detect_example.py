#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import uuid
from pathlib import Path
from urllib import request

from PIL import Image, ImageDraw


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


def call_merge_detect(api_url: str, image_path: Path, timeout_sec: int) -> dict:
    image_bytes = image_path.read_bytes()
    body, content_type = build_multipart_form_data("file", image_path.name, image_bytes)

    req = request.Request(
        url=api_url,
        data=body,
        method="POST",
        headers={"Content-Type": content_type},
    )
    with request.urlopen(req, timeout=timeout_sec) as resp:
        payload = resp.read().decode("utf-8")
    return json.loads(payload)


def draw_boxes(input_image_path: Path, output_image_path: Path, data_items: list[dict]) -> None:
    image = Image.open(input_image_path).convert("RGB")
    drawer = ImageDraw.Draw(image)

    for item in data_items:
        box = item.get("box", [0, 0, 0, 0])
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        drawer.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=3)

    image.save(output_image_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Call /merge_detect and save JSON + boxed image.")
    parser.add_argument("image_path", help="Input image path")
    parser.add_argument(
        "--api-url",
        default="http://localhost:4023/merge_detect",
        help="merge_detect API URL",
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

    result = call_merge_detect(args.api_url, image_path, args.timeout_sec)

    stem = image_path.stem
    json_path = output_dir / f"{stem}_merge_detect.json"
    boxed_image_path = output_dir / f"{stem}_merge_detect_boxes.png"

    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_boxes(image_path, boxed_image_path, result.get("data", []))

    print(json_path)
    print(boxed_image_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

