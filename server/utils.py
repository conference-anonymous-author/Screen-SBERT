import cv2
import numpy as np
import torch
import torchvision


def align_box_points(box: np.ndarray) -> np.ndarray:
    centroid = np.sum(box, axis=0) / 4
    theta = np.arctan2(box[:, 1] - centroid[1], box[:, 0] - centroid[0]) * 180 / np.pi
    indices = np.argsort(theta)
    aligned_box = box[indices]
    start_idx = aligned_box.sum(axis=1).argmin()
    aligned_box = np.roll(aligned_box, 4 - start_idx, 0)
    return aligned_box


def decode_image(image_bytes: bytearray, is_gray: bool = False) -> np.ndarray:
    image_np = np.frombuffer(image_bytes, np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if is_gray else cv2.IMREAD_COLOR
    return cv2.imdecode(image_np, flag)


def od_resize_image(
    img: np.ndarray,
    new_shape: tuple[int, int] = (960, 960),
    auto: bool = False,
    scale_fill: bool = False,
    scaleup: bool = True,
    stride: int = 32,
    center: bool = True,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    shape = img.shape[:2]  # (h, w)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])

    if center:
        dw /= 2
        dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    top = int(round(dh - 0.1)) if center else 0
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1)) if center else 0
    right = int(round(dw + 0.1))

    resized_img = cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    resized_shape = resized_img.shape[:2]
    return resized_img, (shape[1], shape[0]), (resized_shape[1], resized_shape[0])


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    dw = x[..., 2] / 2
    dh = x[..., 3] / 2
    y[..., 0] = x[..., 0] - dw
    y[..., 1] = x[..., 1] - dh
    y[..., 2] = x[..., 0] + dw
    y[..., 3] = x[..., 1] + dh
    return y


def non_max_suppression(
    prediction,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
    max_wh: float = 7680.0,
) -> list[torch.Tensor]:
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    if isinstance(prediction, np.ndarray):
        prediction = torch.from_numpy(prediction)

    if prediction.device.type != "cpu":
        prediction = prediction.cpu()

    assert prediction.ndim == 3, f"expected 3D tensor, got {prediction.shape}"

    prediction = prediction.transpose(-1, -2).float()

    boxes = xywh2xyxy(prediction[..., :4])
    if prediction.shape[-1] > 4:
        class_scores = prediction[..., 4:]
        scores, class_ids = class_scores.max(dim=2)
    else:
        scores = torch.ones(prediction.shape[:2], dtype=prediction.dtype)
        class_ids = torch.zeros(prediction.shape[:2], dtype=torch.long)

    outputs: list[torch.Tensor] = []
    for b in range(prediction.shape[0]):
        mask = scores[b] > conf_thres
        b_boxes = boxes[b][mask]
        b_scores = scores[b][mask]
        b_class_ids = class_ids[b][mask].to(dtype=b_boxes.dtype)

        if b_boxes.numel() == 0:
            outputs.append(torch.zeros((0, 5), dtype=torch.float32))
            continue

        # Class-aware NMS while keeping output schema unchanged.
        nms_boxes = b_boxes + b_class_ids.unsqueeze(1) * max_wh
        keep = torchvision.ops.nms(nms_boxes, b_scores, iou_thres)[:max_det]
        out = torch.cat([b_boxes[keep], b_scores[keep].unsqueeze(1)], dim=1)
        outputs.append(out)

    return outputs


def od_scale_results(
    resized_shape_yx: tuple[int, int, int],
    coords: np.ndarray,
    original_shape_yx: tuple[int, int, int],
    ratio_pad: tuple | None = None,
) -> np.ndarray:
    if ratio_pad is None:
        gain = min(
            resized_shape_yx[0] / original_shape_yx[0],
            resized_shape_yx[1] / original_shape_yx[1],
        )
        pad = (
            (resized_shape_yx[1] - original_shape_yx[1] * gain) / 2,
            (resized_shape_yx[0] - original_shape_yx[0] * gain) / 2,
        )
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain

    coords[:, 0] = np.clip(coords[:, 0], 0, original_shape_yx[1])
    coords[:, 1] = np.clip(coords[:, 1], 0, original_shape_yx[0])
    coords[:, 2] = np.clip(coords[:, 2], 0, original_shape_yx[1])
    coords[:, 3] = np.clip(coords[:, 3], 0, original_shape_yx[0])

    return coords
