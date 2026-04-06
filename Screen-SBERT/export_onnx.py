#!/usr/bin/env python3
import argparse
import inspect
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if TYPE_CHECKING:
    from models import ScreenSBERT


def dynamic_axes():
    return {
        "bbox": {0: "batch", 1: "num_gui"},
        "function_embedding": {0: "batch", 1: "num_gui"},
        "vision_embedding": {0: "batch", 1: "num_gui"},
        "text_embedding": {0: "batch", 1: "num_gui"},
        "padding_mask": {0: "batch", 1: "num_gui"},
        "screen_embedding": {0: "batch"},
    }


def make_dummy_inputs(
    batch_size,
    num_gui,
    device,
    dtype,
    function_dim,
    vision_dim,
    text_dim,
):
    bbox = torch.rand(batch_size, num_gui, 4, device=device, dtype=dtype)
    function_embedding = torch.randn(batch_size, num_gui, function_dim, device=device, dtype=dtype)
    vision_embedding = torch.randn(batch_size, num_gui, vision_dim, device=device, dtype=dtype)
    text_embedding = torch.randn(batch_size, num_gui, text_dim, device=device, dtype=dtype)
    # Keep mask as int32 so TensorRT engine bindings stay simple.
    padding_mask = torch.ones(batch_size, num_gui, device=device, dtype=torch.int32)
    return bbox, function_embedding, vision_embedding, text_embedding, padding_mask


def export_onnx_model(
    model,
    onnx_path,
    input_names,
    output_names,
    batch_size=1,
    num_gui=32,
    opset_version=17,
    use_dynamic_axes=True,
    dtype=torch.float32,
    function_dim=1024,
    vision_dim=768,
    text_dim=1024,
):
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    model_device = next(model.parameters()).device
    dummy_inputs = make_dummy_inputs(
        batch_size=batch_size,
        num_gui=num_gui,
        device=model_device,
        dtype=dtype,
        function_dim=function_dim,
        vision_dim=vision_dim,
        text_dim=text_dim,
    )

    dyn_axes = dynamic_axes() if use_dynamic_axes else None
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_inputs,
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dyn_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
            dynamo=False,
        )


def make_trtexec_command(
    onnx_path,
    plan_path,
    function_dim,
    vision_dim,
    text_dim,
    min_batch=1,
    opt_batch=8,
    max_batch=16,
    min_gui=1,
    opt_gui=32,
    max_gui=128,
    use_fp16=True,
):
    min_shapes = (
        f"bbox:{min_batch}x{min_gui}x4,"
        f"function_embedding:{min_batch}x{min_gui}x{function_dim},"
        f"vision_embedding:{min_batch}x{min_gui}x{vision_dim},"
        f"text_embedding:{min_batch}x{min_gui}x{text_dim},"
        f"padding_mask:{min_batch}x{min_gui}"
    )
    opt_shapes = (
        f"bbox:{opt_batch}x{opt_gui}x4,"
        f"function_embedding:{opt_batch}x{opt_gui}x{function_dim},"
        f"vision_embedding:{opt_batch}x{opt_gui}x{vision_dim},"
        f"text_embedding:{opt_batch}x{opt_gui}x{text_dim},"
        f"padding_mask:{opt_batch}x{opt_gui}"
    )
    max_shapes = (
        f"bbox:{max_batch}x{max_gui}x4,"
        f"function_embedding:{max_batch}x{max_gui}x{function_dim},"
        f"vision_embedding:{max_batch}x{max_gui}x{vision_dim},"
        f"text_embedding:{max_batch}x{max_gui}x{text_dim},"
        f"padding_mask:{max_batch}x{max_gui}"
    )
    fp16_flag = " --fp16" if use_fp16 else ""
    return (
        f"/usr/src/tensorrt/bin/trtexec "
        f"--onnx={onnx_path} "
        f"--saveEngine={plan_path} "
        f"--minShapes={min_shapes} "
        f"--optShapes={opt_shapes} "
        f"--maxShapes={max_shapes}"
        f"{fp16_flag}"
    )


def _sanitize_model_config(model_config: dict, model_cls) -> dict:
    if not isinstance(model_config, dict):
        return {}
    valid_keys = {k for k in inspect.signature(model_cls.__init__).parameters.keys() if k != "self"}
    return {k: v for k, v in model_config.items() if k in valid_keys}


def load_checkpoint(model, ckpt_path: Path) -> None:
    obj = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(obj, dict) and "model_state_dict" in obj:
        state_dict = obj["model_state_dict"]
    elif isinstance(obj, dict):
        state_dict = obj
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(obj)}")
    model.load_state_dict(state_dict, strict=True)


def load_checkpoint_object(ckpt_path: Path) -> dict:
    obj = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(obj)}")
    return obj


def parse_args():
    parser = argparse.ArgumentParser(description="Export ScreenSBERT to ONNX for TensorRT.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained .pt/.pth checkpoint")
    parser.add_argument("--onnx-out", type=Path, required=True, help="Output ONNX path")
    parser.add_argument("--device", type=str, default="cpu", help="Export device (cpu or cuda)")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy batch size used for tracing")
    parser.add_argument("--num-gui", type=int, default=32, help="Dummy GUI token count used for tracing")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--no-dynamic-axes", action="store_true", help="Disable dynamic axes")
    parser.add_argument("--fp16-dummy", action="store_true", help="Use float16 dummy inputs during export")
    return parser.parse_args()


def main():
    from models import ScreenSBERT

    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    ckpt_obj = load_checkpoint_object(args.checkpoint)
    model_config = _sanitize_model_config(ckpt_obj.get("model_config", {}), ScreenSBERT)
    model = ScreenSBERT(**model_config).to(args.device)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    dtype = torch.float16 if args.fp16_dummy else torch.float32
    model.export_onnx(
        onnx_path=args.onnx_out,
        batch_size=args.batch_size,
        num_gui=args.num_gui,
        opset_version=args.opset,
        dynamic_axes=not args.no_dynamic_axes,
        dtype=dtype,
    )

    print(f"Exported ONNX: {args.onnx_out}")
    print("Example trtexec command:")
    print(
        ScreenSBERT.make_trtexec_command(
            onnx_path=str(args.onnx_out),
            plan_path=str(args.onnx_out.with_suffix(".plan")),
            min_batch=1,
            opt_batch=max(1, args.batch_size),
            max_batch=max(1, args.batch_size * 2),
            min_gui=1,
            opt_gui=max(1, args.num_gui),
            max_gui=max(1, args.num_gui * 2),
            use_fp16=True,
        )
    )


if __name__ == "__main__":
    main()
