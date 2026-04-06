from __future__ import annotations

import asyncio
import logging
import os
import time

import numpy as np

from service.postprocess import postprocess_merge_detect, postprocess_parse_gui, postprocess_screen_sbert
from service.preprocess import preprocess_merge_detect, preprocess_parse_gui, preprocess_screen_sbert
from service.utils import run_triton_client


class TritonService:
    def __init__(self, triton_grpc_url: str | None = None):
        self.triton_grpc_url = triton_grpc_url or os.getenv("TRITON_GRPC_URL", "127.0.0.1:4001")

    async def call_merge_detect(
        self,
        image_bytes: bytearray,
    ) -> tuple[list[dict], int, int]:
        t0 = time.time()
        input_dict = preprocess_merge_detect(image_bytes)
        triton_results = await asyncio.to_thread(
            run_triton_client,
            self.triton_grpc_url,
            "merge_detect",
            input_dict,
        )
        results = postprocess_merge_detect(triton_results)
        logging.info(f"merge_detect call Time: {time.time() - t0} at {self.triton_grpc_url}")
        return results

    async def call_parse_gui(
        self,
        image_bytes: bytearray,
    ) -> dict[str, np.ndarray]:
        t0 = time.time()
        input_dict = preprocess_parse_gui(image_bytes)
        triton_results = await asyncio.to_thread(
            run_triton_client,
            self.triton_grpc_url,
            "parse_gui",
            input_dict,
        )
        result = postprocess_parse_gui(triton_results)
        logging.info(f"parse_gui total Time: {time.time() - t0} at {self.triton_grpc_url}")
        return result

    async def call_screen_sbert(
        self,
        image_bytes: bytearray,
    ) -> np.ndarray:
        t0 = time.time()
        input_dict = preprocess_screen_sbert(image_bytes)
        triton_results = await asyncio.to_thread(
            run_triton_client,
            self.triton_grpc_url,
            "screen_sbert",
            input_dict,
        )
        result = postprocess_screen_sbert(triton_results)
        logging.info(f"screen_sbert total Time: {time.time() - t0} at {self.triton_grpc_url}")
        return result
