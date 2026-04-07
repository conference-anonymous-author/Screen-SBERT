from __future__ import annotations

import base64

import numpy as np


def _preprocess_image_bytes(image_bytes: bytearray, output_names: list[str]) -> dict:
    image_base64 = base64.b64encode(image_bytes)
    image_bytes_np = np.array([image_base64], dtype=np.object_)

    return {
        "input_names": ["IMAGE_BYTES"],
        "input_sizes": [[1]],
        "input_data": [image_bytes_np],
        "input_types": ["BYTES"],
        "output_names": output_names,
    }


def preprocess_merge_detect(image_bytes: bytearray) -> dict:
    return _preprocess_image_bytes(
        image_bytes=image_bytes,
        output_names=["OD_DATA_LIST", "ORIGINAL_SHAPE"],
    )


def preprocess_parse_gui(image_bytes: bytearray) -> dict:
    return _preprocess_image_bytes(
        image_bytes=image_bytes,
        output_names=["bbox", "text_embedding", "function_embedding", "vision_embedding"],
    )


def preprocess_screen_sbert(image_bytes: bytearray) -> dict:
    return _preprocess_image_bytes(
        image_bytes=image_bytes,
        output_names=["screen_embedding"],
    )


def preprocess_bge_text_embedding(texts: list[str]) -> dict:
    texts_np = np.array(texts, dtype=np.object_).reshape(-1, 1)
    return {
        "input_names": ["TEXTS"],
        "input_sizes": [list(texts_np.shape)],
        "input_data": [texts_np],
        "input_types": ["BYTES"],
        "output_names": ["EMBEDDINGS"],
    }
