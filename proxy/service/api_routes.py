"""Proxy API routes."""

import io
import logging
import time

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel

from service.TritonService import TritonService

router = APIRouter()
_service = TritonService()


class TextEmbeddingRequest(BaseModel):
    text: str | None = None
    texts: list[str] | None = None


def _normalize_texts(payload: TextEmbeddingRequest) -> list[str]:
    items: list[str] = []
    if payload.text is not None:
        items.append(payload.text)
    if payload.texts is not None:
        items.extend(payload.texts)

    normalized = [x.strip() for x in items if isinstance(x, str) and x.strip()]
    if not normalized:
        raise HTTPException(
            status_code=400,
            detail="Provide non-empty text via 'text' or 'texts'.",
        )
    return normalized


@router.post("/merge_detect", tags=["merge-detection"])
async def merge_detect(file: UploadFile = File(...)):
    t0 = time.time()

    image_bytes = bytearray(await file.read())
    od_data_list, original_width, original_height = await _service.call_merge_detect(image_bytes=image_bytes)

    result_items = []
    for od in od_data_list:
        raw_box = od.get("box", [0, 0, 0, 0])
        caption = str(od.get("caption", "Unknown")).strip()
        if len(caption) == 0:
            caption = "Unknown"
        result_items.append(
            {
                "box": [int(raw_box[0]), int(raw_box[1]), int(raw_box[2]), int(raw_box[3])],
                "text": str(od.get("text", "")),
                "caption": caption,
            }
        )

    logging.info(f"merge_detect time: {time.time() - t0}")
    return {
        "original_width": int(original_width),
        "original_height": int(original_height),
        "data": result_items,
    }


@router.post("/parse_gui", tags=["merge-detection"])
async def parse_gui(
    file: UploadFile = File(...),
    return_type: str = Query(default="npz"),
):
    t0 = time.time()

    image_bytes = bytearray(await file.read())
    result = await _service.call_parse_gui(image_bytes=image_bytes)
    logging.info(f"parse_gui time: {time.time() - t0}")

    return_type = return_type.strip().lower()
    if return_type == "json":
        return {
            "bbox_shape": list(result["bbox"].shape),
            "text_embedding_shape": list(result["text_embedding"].shape),
            "function_embedding_shape": list(result["function_embedding"].shape),
            "vision_embedding_shape": list(result["vision_embedding"].shape),
            "bbox": result["bbox"].astype(np.float32, copy=False).tolist(),
            "text_embedding": result["text_embedding"].astype(np.float32, copy=False).tolist(),
            "function_embedding": result["function_embedding"].astype(np.float32, copy=False).tolist(),
            "vision_embedding": result["vision_embedding"].astype(np.float32, copy=False).tolist(),
        }

    if return_type not in ("npz", "tensor", "numpy"):
        return {
            "error": "invalid_return_type",
            "supported": ["npz", "json"],
        }

    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        bbox=result["bbox"].astype(np.float32, copy=False),
        text_embedding=result["text_embedding"].astype(np.float32, copy=False),
        function_embedding=result["function_embedding"].astype(np.float32, copy=False),
        vision_embedding=result["vision_embedding"].astype(np.float32, copy=False),
    )
    payload = buffer.getvalue()
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="parse_gui.npz"'},
    )


@router.post("/screen_sbert_embed", tags=["screen-sbert"])
async def screen_sbert_embed(
    file: UploadFile = File(...),
    return_type: str = Query(default="json"),
):
    t0 = time.time()

    image_bytes = bytearray(await file.read())
    embedding = await _service.call_screen_sbert(image_bytes=image_bytes)
    logging.info(f"screen_sbert_embed time: {time.time() - t0}")

    return_type = return_type.strip().lower()
    if return_type == "json":
        return {
            "screen_embedding_shape": list(embedding.shape),
            "screen_embedding": embedding.tolist(),
        }

    if return_type not in ("npy", "numpy", "tensor"):
        return {
            "error": "invalid_return_type",
            "supported": ["json", "npy"],
        }

    buffer = io.BytesIO()
    np.save(buffer, embedding.astype(np.float32, copy=False))
    payload = buffer.getvalue()
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="screen_embedding.npy"'},
    )


@router.post("/text_embedding", tags=["text-embedding"])
async def text_embedding(
    payload: TextEmbeddingRequest,
    return_type: str = Query(default="json"),
):
    t0 = time.time()

    texts = _normalize_texts(payload)
    embeddings = await _service.call_bge_text_embedding(texts=texts)
    logging.info(f"text_embedding time: {time.time() - t0}")

    return_type = return_type.strip().lower()
    if return_type == "json":
        return {
            "num_texts": len(texts),
            "embeddings_shape": list(embeddings.shape),
            "embeddings": embeddings.tolist(),
        }

    if return_type not in ("npy", "numpy", "tensor"):
        return {
            "error": "invalid_return_type",
            "supported": ["json", "npy"],
        }

    buffer = io.BytesIO()
    np.save(buffer, embeddings.astype(np.float32, copy=False))
    payload_bytes = buffer.getvalue()
    return Response(
        content=payload_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="text_embeddings.npy"'},
    )
