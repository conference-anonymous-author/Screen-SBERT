#!/bin/bash
set -euo pipefail

echo "Optimization..."
TRTEXEC=/usr/src/tensorrt/bin/trtexec

optimize_if_needed() {
    local model_name="$1"
    local model_dir="$2"
    shift 2

    echo "${model_name}..."

    if [ ! -f "${model_dir}/model.onnx" ]; then
        echo "  Skip: ONNX not found (${model_dir}/model.onnx)"
        return 0
    fi

    if [ -f "${model_dir}/model.plan" ]; then
        echo "  Skip: plan already exists (${model_dir}/model.plan)"
        return 0
    fi

    cd "${model_dir}"
    $TRTEXEC "$@"
}

optimize_if_needed \
    "OCR Text Detection" \
    "/models/ocr_td_engine/1" \
    --onnx=model.onnx \
    --saveEngine=model.plan \
    --minShapes=input:1x3x640x640 \
    --optShapes=input:1x3x2240x640 \
    --maxShapes=input:1x3x2240x2240 \
    --fp16

optimize_if_needed \
    "OCR Text Recognition" \
    "/models/ocr_tr_engine/1" \
    --onnx=model.onnx \
    --saveEngine=model.plan \
    --minShapes=input:1x1x32x256 \
    --optShapes=input:64x1x32x256 \
    --maxShapes=input:128x1x32x256 \
    --fp16

optimize_if_needed \
    "Object Detection" \
    "/models/od_engine/1" \
    --onnx=model.onnx \
    --saveEngine=model.plan

optimize_if_needed \
    "SigLIP" \
    "/models/siglip_engine/1" \
    --onnx=model.onnx \
    --saveEngine=model.plan \
    --minShapes=pixel_values:1x3x224x224 \
    --optShapes=pixel_values:4x3x224x224 \
    --maxShapes=pixel_values:8x3x224x224 \
    --fp16

optimize_if_needed \
    "BGE-M3" \
    "/models/bge_m3_engine/1" \
    --onnx=model.onnx \
    --saveEngine=model.plan \
    --minShapes=input_ids:1x1,attention_mask:1x1 \
    --optShapes=input_ids:8x64,attention_mask:8x64 \
    --maxShapes=input_ids:16x128,attention_mask:16x128 \
    --fp16

optimize_if_needed \
    "Screen-SBERT" \
    "/models/screen_sbert_engine/1" \
    --onnx=model.onnx \
    --saveEngine=model.plan \
    --minShapes=bbox:1x1x4,function_embedding:1x1x1024,vision_embedding:1x1x768,text_embedding:1x1x1024,padding_mask:1x1 \
    --optShapes=bbox:1x64x4,function_embedding:1x64x1024,vision_embedding:1x64x768,text_embedding:1x64x1024,padding_mask:1x64 \
    --maxShapes=bbox:4x128x4,function_embedding:4x128x1024,vision_embedding:4x128x768,text_embedding:4x128x1024,padding_mask:4x128

echo "Done."
