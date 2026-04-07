from __future__ import annotations

import json

import numpy as np


def postprocess_merge_detect(triton_results: dict) -> tuple[list[dict], int, int]:
    original_width, original_height = triton_results["ORIGINAL_SHAPE"].tolist()
    od_data_list_raw = triton_results["OD_DATA_LIST"].item().decode("utf-8")
    od_data_list = json.loads(od_data_list_raw)
    return od_data_list, int(original_width), int(original_height)


def postprocess_parse_gui(triton_results: dict) -> dict[str, np.ndarray]:
    return {
        "bbox": triton_results["bbox"].astype(np.float32, copy=False),
        "text_embedding": triton_results["text_embedding"].astype(np.float32, copy=False),
        "function_embedding": triton_results["function_embedding"].astype(np.float32, copy=False),
        "vision_embedding": triton_results["vision_embedding"].astype(np.float32, copy=False),
    }


def postprocess_screen_sbert(triton_results: dict) -> np.ndarray:
    return triton_results["screen_embedding"].astype(np.float32, copy=False)


def postprocess_bge_text_embedding(triton_results: dict) -> np.ndarray:
    return triton_results["EMBEDDINGS"].astype(np.float32, copy=False)
