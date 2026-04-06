import argparse
import gc
import json
import math
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from models import ScreenSBERT


REQUIRED_FEATURE_KEYS = (
    "bbox",
    "text_embedding",
    "function_embedding",
    "vision_embedding",
)
DEFAULT_SPLIT_TRAIN = "train"
DEFAULT_SPLIT_VAL_CANDIDATES = ("validation", "val")
DEFAULT_SPLIT_TEST = "test"


@dataclass(frozen=True)
class Episode:
    app_name: str
    class_name: str
    anchor_path: Path
    positive_path: Path
    negative_paths: Tuple[Path, ...]


@dataclass
class TrainResult:
    best_epoch: int
    best_score: float
    best_threshold: float
    best_metrics: Dict[str, float]
    optuna_objective_score: float

def apply_training_defaults_from_checkpoint(args) -> None:
    if not args.training_defaults_checkpoint:
        return

    ckpt_path = Path(args.training_defaults_checkpoint).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"training defaults checkpoint not found: {ckpt_path}")

    obj = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(obj)}")

    model_cfg = obj.get("model_config", {})
    train_cfg = obj.get("train_config", {})
    if not isinstance(model_cfg, dict) or not isinstance(train_cfg, dict):
        raise ValueError(f"Invalid checkpoint config layout in: {ckpt_path}")

    model_keys = [
        "embed_dim",
        "num_heads",
        "num_layers",
        "d_ff",
        "dropout",
        "attn_dropout",
        "layer_scale_init",
        "width",
        "height",
        "num_buckets",
        "max_distance",
        "log_base",
        "function_proj_hidden_dim",
        "vision_proj_hidden_dim",
        "text_proj_hidden_dim",
        "proj_dropout",
        "proj_init_scale",
    ]
    train_keys = [
        "epochs",
        "lr",
        "weight_decay",
        "temperature",
        "contrastive_margin",
        "contrastive_margin_weight",
        "grad_clip_norm",
        "early_stopping_patience",
        "train_classes_per_app",
        "episodes_per_app",
    ]

    for k in model_keys:
        if k in model_cfg and hasattr(args, k):
            setattr(args, k, model_cfg[k])
    for k in train_keys:
        if k in train_cfg and hasattr(args, k):
            setattr(args, k, train_cfg[k])

    print(f"[CONFIG] loaded model/train defaults from checkpoint: {ckpt_path}", flush=True)


class TeeTextIO:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    @property
    def encoding(self):
        if self.streams:
            return getattr(self.streams[0], "encoding", "utf-8")
        return "utf-8"

    def isatty(self):
        return False


@contextmanager
def trial_log_capture(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as trial_f:
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        sys.stdout = TeeTextIO(orig_stdout, trial_f)
        sys.stderr = TeeTextIO(orig_stderr, trial_f)
        try:
            yield
        finally:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                sys.stdout = orig_stdout
                sys.stderr = orig_stderr


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_csv_arg(value: str) -> List[str]:
    if not value:
        return []
    return [token.strip() for token in value.split(",") if token.strip()]


def _is_sample_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return all((path / f"{k}.npy").is_file() for k in REQUIRED_FEATURE_KEYS)


def discover_dataset_index(dataset_root: Path) -> Dict[str, Dict[str, List[Path]]]:
    """
    Expects dataset layout:
      dataset_root/
        app_name/
          class_name/
            *.npz
            .../parse_gui.npz
            sample_idx/
              bbox.npy
              text_embedding.npy
              function_embedding.npy
              vision_embedding.npy
    """
    index: Dict[str, Dict[str, List[Path]]] = {}

    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")

    for app_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        class_map: Dict[str, List[Path]] = {}
        for class_dir in sorted(p for p in app_dir.iterdir() if p.is_dir()):
            sample_dirs = [p.resolve() for p in class_dir.iterdir() if _is_sample_dir(p)]
            direct_npz = [p.resolve() for p in class_dir.glob("*.npz")]
            nested_npz = [p.resolve() for p in class_dir.rglob("parse_gui.npz")]
            paths = sorted({*sample_dirs, *direct_npz, *nested_npz})
            if paths:
                class_map[class_dir.name] = paths

        if class_map:
            index[app_dir.name] = class_map

    if not index:
        raise RuntimeError(f"No samples found under: {dataset_root}")

    return index


def _filter_apps(
    split_index: Dict[str, Dict[str, List[Path]]],
    apps: Sequence[str],
    split_name: str,
) -> Dict[str, Dict[str, List[Path]]]:
    if not apps:
        return split_index

    missing = [app for app in apps if app not in split_index]
    if missing:
        raise ValueError(f"{split_name}: unknown apps {missing}")
    return {app: split_index[app] for app in apps}


def _discover_predefined_split_dirs(dataset_root: Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    train_dir = dataset_root / DEFAULT_SPLIT_TRAIN
    if not train_dir.is_dir():
        return None, None, None

    val_dir = None
    for candidate in DEFAULT_SPLIT_VAL_CANDIDATES:
        p = dataset_root / candidate
        if p.is_dir():
            val_dir = p
            break
    if val_dir is None:
        return None, None, None

    test_dir = dataset_root / DEFAULT_SPLIT_TEST
    if not test_dir.is_dir():
        test_dir = None

    return train_dir, val_dir, test_dir


def build_dataset_splits(
    dataset_root: Path,
    train_apps: Sequence[str],
    ood_apps: Sequence[str],
    train_classes_per_app: int,
    seed: int,
) -> Tuple[
    Dict[str, Dict[str, List[Path]]],
    Dict[str, Dict[str, List[Path]]],
    Dict[str, Dict[str, List[Path]]],
    str,
]:
    train_dir, val_dir, test_dir = _discover_predefined_split_dirs(dataset_root)
    if train_dir is not None and val_dir is not None:
        train_split = discover_dataset_index(train_dir)
        val_split = discover_dataset_index(val_dir)
        ood_split: Dict[str, Dict[str, List[Path]]] = {}
        if test_dir is not None:
            ood_split = discover_dataset_index(test_dir)

        if train_apps:
            train_split = _filter_apps(train_split, train_apps, "train split")
            val_split = _filter_apps(val_split, train_apps, "validation split")

        if ood_apps:
            ood_split = _filter_apps(ood_split, ood_apps, "test split")

        return train_split, val_split, ood_split, "predefined"

    full_index = discover_dataset_index(dataset_root)
    train_split, val_split, ood_split = split_train_val_apps(
        full_index=full_index,
        train_apps=train_apps,
        ood_apps=ood_apps,
        train_classes_per_app=train_classes_per_app,
        seed=seed,
    )
    return train_split, val_split, ood_split, "random_class_split"


def split_train_val_apps(
    full_index: Dict[str, Dict[str, List[Path]]],
    train_apps: Sequence[str],
    ood_apps: Sequence[str],
    train_classes_per_app: int,
    seed: int,
) -> Tuple[
    Dict[str, Dict[str, List[Path]]],
    Dict[str, Dict[str, List[Path]]],
    Dict[str, Dict[str, List[Path]]],
]:
    all_apps = sorted(full_index.keys())

    if train_apps:
        missing = [a for a in train_apps if a not in full_index]
        if missing:
            raise ValueError(f"Unknown train apps: {missing}")
        selected_train_apps = list(train_apps)
    else:
        selected_train_apps = [a for a in all_apps if a not in set(ood_apps)]

    if not selected_train_apps:
        raise ValueError("No train apps selected.")

    selected_ood_apps = list(ood_apps)
    missing_ood = [a for a in selected_ood_apps if a not in full_index]
    if missing_ood:
        raise ValueError(f"Unknown OOD apps: {missing_ood}")

    rng = random.Random(seed)

    train_split: Dict[str, Dict[str, List[Path]]] = {}
    val_split: Dict[str, Dict[str, List[Path]]] = {}
    ood_split: Dict[str, Dict[str, List[Path]]] = {}

    for app in selected_train_apps:
        class_map = full_index[app]
        class_names = sorted(class_map.keys())
        if len(class_names) < 2:
            raise ValueError(f"App '{app}' must have at least 2 classes for contrastive training.")

        shuffled = class_names[:]
        rng.shuffle(shuffled)

        if train_classes_per_app <= 0 or train_classes_per_app >= len(shuffled):
            raise ValueError(
                f"For app '{app}', train_classes_per_app must be in [1, {len(shuffled)-1}] "
                f"but got {train_classes_per_app}"
            )

        train_classes = sorted(shuffled[:train_classes_per_app])
        val_classes = sorted(shuffled[train_classes_per_app:])

        train_split[app] = {c: class_map[c] for c in train_classes}
        val_split[app] = {c: class_map[c] for c in val_classes}

    for app in selected_ood_apps:
        ood_split[app] = {c: samples[:] for c, samples in full_index[app].items()}

    return train_split, val_split, ood_split


def validate_split_requirements(split_index: Dict[str, Dict[str, List[Path]]], split_name: str) -> None:
    for app, class_map in split_index.items():
        if len(class_map) < 2:
            raise ValueError(f"{split_name} app '{app}' must contain at least 2 classes.")
        for class_name, samples in class_map.items():
            if len(samples) < 2:
                raise ValueError(
                    f"{split_name} app '{app}' class '{class_name}' must contain at least 2 samples "
                    f"(got {len(samples)})."
                )


def load_single_feature(path: Path) -> Dict[str, torch.Tensor]:
    if path.is_dir():
        missing = [k for k in REQUIRED_FEATURE_KEYS if not (path / f"{k}.npy").is_file()]
        if missing:
            raise KeyError(f"Missing npy keys {missing} in {path}")
        bbox = np.load(path / "bbox.npy").astype(np.float32)
        text_embedding = np.load(path / "text_embedding.npy").astype(np.float32)
        function_embedding = np.load(path / "function_embedding.npy").astype(np.float32)
        vision_embedding = np.load(path / "vision_embedding.npy").astype(np.float32)
    else:
        data = np.load(path)
        missing = [k for k in REQUIRED_FEATURE_KEYS if k not in data.files]
        if missing:
            raise KeyError(f"Missing keys {missing} in {path}")
        bbox = data["bbox"].astype(np.float32)
        text_embedding = data["text_embedding"].astype(np.float32)
        function_embedding = data["function_embedding"].astype(np.float32)
        vision_embedding = data["vision_embedding"].astype(np.float32)

    n = bbox.shape[0]
    if bbox.ndim != 2 or bbox.shape[1] != 4:
        raise ValueError(f"bbox must have shape [N,4], got {bbox.shape} in {path}")
    if text_embedding.ndim != 2 or text_embedding.shape[0] != n or text_embedding.shape[1] != 1024:
        raise ValueError(f"text_embedding must have shape [N,1024], got {text_embedding.shape} in {path}")
    if function_embedding.ndim != 2 or function_embedding.shape[0] != n or function_embedding.shape[1] != 1024:
        raise ValueError(
            f"function_embedding must have shape [N,1024], got {function_embedding.shape} in {path}"
        )
    if vision_embedding.ndim != 2 or vision_embedding.shape[0] != n or vision_embedding.shape[1] != 768:
        raise ValueError(f"vision_embedding must have shape [N,768], got {vision_embedding.shape} in {path}")

    return {
        "bbox": torch.from_numpy(bbox),
        "text_embedding": torch.from_numpy(text_embedding),
        "function_embedding": torch.from_numpy(function_embedding),
        "vision_embedding": torch.from_numpy(vision_embedding),
    }


def preload_feature_store(
    split_indexes: Iterable[Dict[str, Dict[str, List[Path]]]],
) -> Dict[Path, Dict[str, torch.Tensor]]:
    unique_paths: List[Path] = []
    seen = set()
    for split in split_indexes:
        for class_map in split.values():
            for sample_paths in class_map.values():
                for p in sample_paths:
                    rp = p.resolve()
                    if rp not in seen:
                        seen.add(rp)
                        unique_paths.append(rp)

    store: Dict[Path, Dict[str, torch.Tensor]] = {}
    for p in unique_paths:
        store[p] = load_single_feature(p)
    return store


def collate_screen_batch(
    sample_paths: Sequence[Path],
    feature_store: Dict[Path, Dict[str, torch.Tensor]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bboxes = []
    text_embeddings = []
    function_embeddings = []
    vision_embeddings = []
    lengths = []

    for path in sample_paths:
        item = feature_store[path.resolve()]
        bboxes.append(item["bbox"])
        text_embeddings.append(item["text_embedding"])
        function_embeddings.append(item["function_embedding"])
        vision_embeddings.append(item["vision_embedding"])
        lengths.append(item["bbox"].shape[0])

    padded_bbox = pad_sequence(bboxes, batch_first=True).to(device=device, dtype=torch.float32)
    padded_text = pad_sequence(text_embeddings, batch_first=True).to(device=device, dtype=torch.float32)
    padded_function = pad_sequence(function_embeddings, batch_first=True).to(device=device, dtype=torch.float32)
    padded_vision = pad_sequence(vision_embeddings, batch_first=True).to(device=device, dtype=torch.float32)

    max_len = padded_bbox.shape[1]
    mask = torch.zeros((len(sample_paths), max_len), dtype=torch.int32, device=device)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1

    return padded_bbox, padded_function, padded_vision, padded_text, mask


def encode_screens(
    model: nn.Module,
    sample_paths: Sequence[Path],
    feature_store: Dict[Path, Dict[str, torch.Tensor]],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    all_embeddings = []
    for start in range(0, len(sample_paths), batch_size):
        batch_paths = sample_paths[start : start + batch_size]
        bbox, function_emb, vision_emb, text_emb, padding_mask = collate_screen_batch(
            batch_paths, feature_store, device
        )
        with torch.no_grad():
            screen_emb = model(bbox, function_emb, vision_emb, text_emb, padding_mask)
            screen_emb = F.normalize(screen_emb, dim=-1)
        all_embeddings.append(screen_emb)
    return torch.cat(all_embeddings, dim=0)


def build_train_episodes(
    train_split: Dict[str, Dict[str, List[Path]]],
    rng: random.Random,
    episodes_per_app: int,
) -> List[Episode]:
    episodes: List[Episode] = []

    for app_name, class_map in train_split.items():
        class_names = sorted(class_map.keys())
        app_episodes: List[Episode] = []

        for cls in class_names:
            anchor_candidates = class_map[cls]
            for anchor in anchor_candidates:
                positive_pool = [p for p in class_map[cls] if p != anchor]
                if not positive_pool:
                    continue
                positive = rng.choice(positive_pool)

                negatives = []
                for neg_cls in class_names:
                    if neg_cls == cls:
                        continue
                    negatives.append(rng.choice(class_map[neg_cls]))

                app_episodes.append(
                    Episode(
                        app_name=app_name,
                        class_name=cls,
                        anchor_path=anchor,
                        positive_path=positive,
                        negative_paths=tuple(negatives),
                    )
                )

        if episodes_per_app > 0 and len(app_episodes) > episodes_per_app:
            app_episodes = rng.sample(app_episodes, k=episodes_per_app)

        episodes.extend(app_episodes)

    rng.shuffle(episodes)
    return episodes


def contrastive_softmax_loss(
    anchor_embedding: torch.Tensor,
    positive_embedding: torch.Tensor,
    negative_embeddings: torch.Tensor,
    temperature: float,
    contrastive_margin: float = 0.1,
    contrastive_margin_weight: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      loss, similarities where similarities = [sim_pos, sim_neg_1, ...]
    """
    if negative_embeddings.ndim != 2 or negative_embeddings.shape[0] < 1:
        raise ValueError("negative_embeddings must have shape [K, D] with K >= 1")

    sim_pos = F.cosine_similarity(anchor_embedding.unsqueeze(0), positive_embedding.unsqueeze(0), dim=-1)
    sim_neg = F.cosine_similarity(anchor_embedding.unsqueeze(0), negative_embeddings, dim=-1)

    sims = torch.cat([sim_pos, sim_neg], dim=0)
    logits = sims / temperature
    target = torch.zeros((1,), dtype=torch.long, device=logits.device)
    ce_loss = F.cross_entropy(logits.unsqueeze(0), target)

    if contrastive_margin > 0.0 and contrastive_margin_weight > 0.0:
        # Enforce sim_pos - sim_neg >= margin for all negatives.
        margin_violation = torch.relu(contrastive_margin - (sim_pos.unsqueeze(0) - sim_neg))
        margin_loss = margin_violation.mean()
    else:
        margin_loss = torch.zeros((), dtype=ce_loss.dtype, device=ce_loss.device)

    loss = ce_loss + contrastive_margin_weight * margin_loss
    return loss, sims.detach()


def compute_optuna_objective_score(
    val_scores: Sequence[float],
    mode: str,
    last_k: int,
    trend_alpha: float,
) -> float:
    if len(val_scores) == 0:
        return 0.0

    if mode == "best":
        return float(max(val_scores))

    k = max(1, min(last_k, len(val_scores)))
    window = [float(v) for v in val_scores[-k:]]
    mean_last_k = float(sum(window) / len(window))

    if mode == "last_k_mean":
        return mean_last_k

    if mode == "trend":
        if len(window) < 2:
            slope = 0.0
        else:
            x_mean = (len(window) - 1) / 2.0
            y_mean = mean_last_k
            num = 0.0
            den = 0.0
            for i, y in enumerate(window):
                dx = float(i) - x_mean
                num += dx * (y - y_mean)
                den += dx * dx
            slope = num / den if den > 0.0 else 0.0
        return float(mean_last_k + trend_alpha * slope)

    raise ValueError(f"Unknown optuna objective mode: {mode}")


def collect_pos_neg_similarities(
    embeddings_by_app_class: Dict[str, Dict[str, torch.Tensor]],
) -> Tuple[np.ndarray, np.ndarray]:
    pos_sims: List[float] = []
    neg_sims: List[float] = []

    for _, class_embeddings in embeddings_by_app_class.items():
        classes = sorted(class_embeddings.keys())

        for cls in classes:
            emb = F.normalize(class_embeddings[cls], dim=-1)
            n = emb.shape[0]
            if n < 2:
                continue
            sim_mat = emb @ emb.t()
            tri = torch.triu_indices(n, n, offset=1)
            pos_vals = sim_mat[tri[0], tri[1]].detach().cpu().numpy().tolist()
            pos_sims.extend(pos_vals)

        for cls_a, cls_b in combinations(classes, 2):
            emb_a = F.normalize(class_embeddings[cls_a], dim=-1)
            emb_b = F.normalize(class_embeddings[cls_b], dim=-1)
            cross_sim = (emb_a @ emb_b.t()).reshape(-1)
            neg_sims.extend(cross_sim.detach().cpu().numpy().tolist())

    if not pos_sims or not neg_sims:
        raise RuntimeError("Validation similarity collection failed: empty pos or neg similarity set.")

    return np.asarray(pos_sims, dtype=np.float32), np.asarray(neg_sims, dtype=np.float32)


def _binary_metrics_from_counts(tp: int, fn: int, fp: int, tn: int) -> Dict[str, float]:
    total_pos = tp + fn
    total_neg = tn + fp

    tpr = tp / total_pos if total_pos > 0 else 0.0
    fnr = fn / total_pos if total_pos > 0 else 0.0
    tnr = tn / total_neg if total_neg > 0 else 0.0
    fpr = fp / total_neg if total_neg > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    pos_f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    neg_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    neg_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    neg_f1 = (
        2.0 * neg_precision * neg_recall / (neg_precision + neg_recall)
        if (neg_precision + neg_recall) > 0
        else 0.0
    )
    macro_f1 = 0.5 * (pos_f1 + neg_f1)
    macro_precision = 0.5 * (precision + neg_precision)
    macro_recall = 0.5 * (recall + neg_recall)
    balanced_accuracy = 0.5 * (tpr + tnr)
    accuracy = (tp + tn) / (total_pos + total_neg) if (total_pos + total_neg) > 0 else 0.0

    return {
        "tpr": float(tpr),
        "fnr": float(fnr),
        "tnr": float(tnr),
        "fpr": float(fpr),
        "precision": float(precision),
        "recall": float(recall),
        "pos_f1": float(pos_f1),
        "neg_f1": float(neg_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_accuracy),
        "accuracy": float(accuracy),
    }


def calibrate_threshold(
    pos_sims: np.ndarray,
    neg_sims: np.ndarray,
    max_threshold_points: int = 2001,
) -> Dict[str, float]:
    all_sims = np.concatenate([pos_sims, neg_sims], axis=0)

    uniq = np.unique(all_sims)
    if uniq.size > max_threshold_points:
        thresholds = np.quantile(all_sims, np.linspace(0.0, 1.0, num=max_threshold_points))
    else:
        thresholds = uniq

    best_metrics: Optional[Dict[str, float]] = None
    best_threshold = float(thresholds[0])

    best_bal_acc = -1.0
    best_macro_f1 = -1.0
    best_separation = -1.0

    best_eer_gap = math.inf
    eer_threshold = float(thresholds[0])
    eer_fpr = 1.0
    eer_fnr = 1.0

    for threshold in thresholds:
        pos_pred = pos_sims >= threshold
        neg_pred = neg_sims >= threshold

        tp = int(pos_pred.sum())
        fn = int((~pos_pred).sum())
        fp = int(neg_pred.sum())
        tn = int((~neg_pred).sum())

        metrics = _binary_metrics_from_counts(tp, fn, fp, tn)
        separation = metrics["tpr"] - metrics["fpr"]

        if (
            metrics["balanced_accuracy"] > best_bal_acc
            or (
                math.isclose(metrics["balanced_accuracy"], best_bal_acc, rel_tol=1e-9, abs_tol=1e-9)
                and metrics["macro_f1"] > best_macro_f1
            )
            or (
                math.isclose(metrics["balanced_accuracy"], best_bal_acc, rel_tol=1e-9, abs_tol=1e-9)
                and math.isclose(metrics["macro_f1"], best_macro_f1, rel_tol=1e-9, abs_tol=1e-9)
                and separation > best_separation
            )
        ):
            best_bal_acc = metrics["balanced_accuracy"]
            best_macro_f1 = metrics["macro_f1"]
            best_separation = separation
            best_threshold = float(threshold)
            best_metrics = metrics

        eer_gap = abs(metrics["fpr"] - metrics["fnr"])
        if eer_gap < best_eer_gap:
            best_eer_gap = eer_gap
            eer_threshold = float(threshold)
            eer_fpr = metrics["fpr"]
            eer_fnr = metrics["fnr"]

    assert best_metrics is not None

    result = {
        "threshold": best_threshold,
        "eer_threshold": eer_threshold,
        "eer_fpr": float(eer_fpr),
        "eer_fnr": float(eer_fnr),
        "pos_mean": float(pos_sims.mean()),
        "pos_std": float(pos_sims.std()),
        "neg_mean": float(neg_sims.mean()),
        "neg_std": float(neg_sims.std()),
        "pos_count": float(pos_sims.shape[0]),
        "neg_count": float(neg_sims.shape[0]),
        **best_metrics,
    }
    return result


def compute_embeddings_by_app_class(
    model: nn.Module,
    split_index: Dict[str, Dict[str, List[Path]]],
    feature_store: Dict[Path, Dict[str, torch.Tensor]],
    device: torch.device,
    eval_batch_size: int,
) -> Dict[str, Dict[str, torch.Tensor]]:
    model.eval()
    embeddings: Dict[str, Dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        for app, class_map in split_index.items():
            embeddings[app] = {}
            for cls, sample_paths in class_map.items():
                emb = encode_screens(
                    model=model,
                    sample_paths=sample_paths,
                    feature_store=feature_store,
                    device=device,
                    batch_size=eval_batch_size,
                )
                embeddings[app][cls] = emb
    return embeddings


def evaluate_split_with_calibration(
    model: nn.Module,
    split_index: Dict[str, Dict[str, List[Path]]],
    feature_store: Dict[Path, Dict[str, torch.Tensor]],
    device: torch.device,
    eval_batch_size: int,
) -> Dict[str, float]:
    embeddings_by_app_class = compute_embeddings_by_app_class(
        model=model,
        split_index=split_index,
        feature_store=feature_store,
        device=device,
        eval_batch_size=eval_batch_size,
    )
    pos_sims, neg_sims = collect_pos_neg_similarities(embeddings_by_app_class)
    return calibrate_threshold(pos_sims, neg_sims)


def evaluate_split_at_threshold(
    model: nn.Module,
    split_index: Dict[str, Dict[str, List[Path]]],
    feature_store: Dict[Path, Dict[str, torch.Tensor]],
    device: torch.device,
    eval_batch_size: int,
    threshold: float,
) -> Dict[str, float]:
    embeddings_by_app_class = compute_embeddings_by_app_class(
        model=model,
        split_index=split_index,
        feature_store=feature_store,
        device=device,
        eval_batch_size=eval_batch_size,
    )
    pos_sims, neg_sims = collect_pos_neg_similarities(embeddings_by_app_class)

    tp = int((pos_sims >= threshold).sum())
    fn = int((pos_sims < threshold).sum())
    fp = int((neg_sims >= threshold).sum())
    tn = int((neg_sims < threshold).sum())

    metrics = _binary_metrics_from_counts(tp, fn, fp, tn)
    metrics.update(
        {
            "threshold": float(threshold),
            "pos_mean": float(pos_sims.mean()),
            "pos_std": float(pos_sims.std()),
            "neg_mean": float(neg_sims.mean()),
            "neg_std": float(neg_sims.std()),
            "pos_count": float(pos_sims.shape[0]),
            "neg_count": float(neg_sims.shape[0]),
        }
    )
    return metrics


def train_one_run(
    model: nn.Module,
    train_split: Dict[str, Dict[str, List[Path]]],
    val_split: Dict[str, Dict[str, List[Path]]],
    feature_store: Dict[Path, Dict[str, torch.Tensor]],
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    temperature: float,
    contrastive_margin: float,
    contrastive_margin_weight: float,
    grad_clip_norm: float,
    eval_batch_size: int,
    episodes_per_app: int,
    log_interval_steps: int,
    early_stopping_patience: int,
    seed: int,
    trial=None,
) -> TrainResult:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_score = -1.0
    best_threshold = 0.0
    best_metrics: Dict[str, float] = {}
    best_epoch = -1
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    no_improve_streak = 0

    rng = random.Random(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        episodes = build_train_episodes(train_split=train_split, rng=rng, episodes_per_app=episodes_per_app)
        if not episodes:
            raise RuntimeError("No training episodes generated.")

        epoch_loss = 0.0
        epoch_pos_sim = 0.0
        epoch_neg_sim = 0.0

        optimizer.zero_grad(set_to_none=True)

        num_episodes = len(episodes)
        for step_idx, ep in enumerate(episodes, start=1):
            anchor_batch = [ep.anchor_path]
            candidate_batch = [ep.positive_path, *ep.negative_paths]

            bbox_a, function_a, vision_a, text_a, mask_a = collate_screen_batch(anchor_batch, feature_store, device)
            bbox_b, function_b, vision_b, text_b, mask_b = collate_screen_batch(candidate_batch, feature_store, device)

            anchor_emb = model(bbox_a, function_a, vision_a, text_a, mask_a).squeeze(0)
            candidate_emb = model(bbox_b, function_b, vision_b, text_b, mask_b)

            anchor_emb = F.normalize(anchor_emb, dim=-1)
            candidate_emb = F.normalize(candidate_emb, dim=-1)

            positive_emb = candidate_emb[0]
            negative_embs = candidate_emb[1:]

            loss, sims = contrastive_softmax_loss(
                anchor_embedding=anchor_emb,
                positive_embedding=positive_emb,
                negative_embeddings=negative_embs,
                temperature=temperature,
                contrastive_margin=contrastive_margin,
                contrastive_margin_weight=contrastive_margin_weight,
            )

            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += float(loss.item())
            epoch_pos_sim += float(sims[0].item())
            if sims.numel() > 1:
                epoch_neg_sim += float(sims[1:].mean().item())

        avg_loss = epoch_loss / num_episodes
        avg_pos_sim = epoch_pos_sim / num_episodes
        avg_neg_sim = epoch_neg_sim / num_episodes

        train_calib_metrics = evaluate_split_with_calibration(
            model=model,
            split_index=train_split,
            feature_store=feature_store,
            device=device,
            eval_batch_size=eval_batch_size,
        )
        eval_threshold = float(train_calib_metrics["threshold"])
        val_metrics = evaluate_split_at_threshold(
            model=model,
            split_index=val_split,
            feature_store=feature_store,
            device=device,
            eval_batch_size=eval_batch_size,
            threshold=eval_threshold,
        )
        val_score = float(val_metrics["balanced_accuracy"])

        trial_prefix = f"[Trial {trial.number:03d}] " if trial is not None else ""

        # Epoch summary table (header shown every epoch for easier log scanning).
        print(
            f"{trial_prefix}"
            f"{'epoch':>5} | {'loss':>10} | {'tr_pos':>8} | {'tr_neg':>8}",
            flush=True,
        )
        print(
            f"{trial_prefix}"
            f"{epoch:>5d} | {avg_loss:>10.6f} | {avg_pos_sim:>8.4f} | {avg_neg_sim:>8.4f}",
            flush=True,
        )

        # Detailed metric table (header shown every epoch for consistency).
        print(
            f"{trial_prefix}"
            f"{'split':<11} | {'bal_acc':>8} | {'macro_f1':>9} | {'macro_precision':>15} | "
            f"{'macro_recall':>12} | {'threshold':>9} | {'pos_mean':>8} | {'neg_mean':>8}",
            flush=True,
        )
        print(
            f"{trial_prefix}"
            f"{'train_calib':<11} | {train_calib_metrics['balanced_accuracy']:>8.4f} | "
            f"{train_calib_metrics['macro_f1']:>9.4f} | {train_calib_metrics['macro_precision']:>15.4f} | "
            f"{train_calib_metrics['macro_recall']:>12.4f} | {eval_threshold:>9.4f} | "
            f"{train_calib_metrics['pos_mean']:>8.4f} | {train_calib_metrics['neg_mean']:>8.4f}",
            flush=True,
        )
        print(
            f"{trial_prefix}"
            f"{'val':<11} | {val_metrics['balanced_accuracy']:>8.4f} | "
            f"{val_metrics['macro_f1']:>9.4f} | {val_metrics['macro_precision']:>15.4f} | "
            f"{val_metrics['macro_recall']:>12.4f} | {val_metrics['threshold']:>9.4f} | "
            f"{val_metrics['pos_mean']:>8.4f} | {val_metrics['neg_mean']:>8.4f}",
            flush=True,
        )

        if val_score > best_score:
            best_score = val_score
            best_threshold = eval_threshold
            best_metrics = dict(val_metrics)
            best_metrics["train_calib_threshold"] = eval_threshold
            best_metrics["train_calib_balanced_accuracy"] = float(train_calib_metrics["balanced_accuracy"])
            best_metrics["train_calib_macro_f1"] = float(train_calib_metrics["macro_f1"])
            best_epoch = epoch
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve_streak = 0
        else:
            no_improve_streak += 1

        if trial is not None:
            trial.set_user_attr("epochs_completed", int(epoch))

        if early_stopping_patience > 0 and no_improve_streak >= early_stopping_patience:
            print(
                f"{trial_prefix}EARLY_STOP: no val_bal_acc improvement for {no_improve_streak} "
                f"epoch(s) (patience={early_stopping_patience}), best_epoch={best_epoch}",
                flush=True,
            )
            if trial is not None:
                trial.set_user_attr("early_stopped", True)
                trial.set_user_attr("early_stop_epoch", int(epoch))
                trial.set_user_attr("early_stop_no_improve_streak", int(no_improve_streak))
            break

    if best_epoch < 0:
        raise RuntimeError("Training finished without validation result.")

    # Ensure caller sees the best validation checkpoint weights.
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict, strict=True)

    # Optuna objective score is validation-based.
    optuna_objective_score = best_score

    return TrainResult(
        best_epoch=best_epoch,
        best_score=best_score,
        best_threshold=best_threshold,
        best_metrics=best_metrics,
        optuna_objective_score=optuna_objective_score,
    )


def build_model_from_args(args, trial=None) -> ScreenSBERT:
    if trial is not None:
        return ScreenSBERT.from_optuna_trial(trial)

    return ScreenSBERT(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        layer_scale_init=args.layer_scale_init,
        width=args.width,
        height=args.height,
        num_buckets=args.num_buckets,
        max_distance=args.max_distance,
        log_base=args.log_base,
        function_proj_hidden_dim=args.function_proj_hidden_dim,
        vision_proj_hidden_dim=args.vision_proj_hidden_dim,
        text_proj_hidden_dim=args.text_proj_hidden_dim,
        proj_dropout=args.proj_dropout,
        proj_init_scale=args.proj_init_scale,
    )


def run_train(args) -> None:
    set_seed(args.seed)

    dataset_root = Path(args.dataset_root).resolve()
    train_apps = parse_csv_arg(args.train_apps)
    ood_apps = parse_csv_arg(args.ood_apps)
    train_split, val_split, ood_split, split_mode = build_dataset_splits(
        dataset_root=dataset_root,
        train_apps=train_apps,
        ood_apps=ood_apps,
        train_classes_per_app=args.train_classes_per_app,
        seed=args.seed,
    )

    print(
        f"[DATA] split_mode={split_mode}, train_apps={len(train_split)}, "
        f"val_apps={len(val_split)}, ood_apps={len(ood_split)}",
        flush=True,
    )

    validate_split_requirements(train_split, split_name="train")
    validate_split_requirements(val_split, split_name="val")

    feature_store = preload_feature_store([train_split, val_split, ood_split])

    device = torch.device(args.device)
    model = build_model_from_args(args).to(device)

    result = train_one_run(
        model=model,
        train_split=train_split,
        val_split=val_split,
        feature_store=feature_store,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        contrastive_margin=args.contrastive_margin,
        contrastive_margin_weight=args.contrastive_margin_weight,
        grad_clip_norm=args.grad_clip_norm,
        eval_batch_size=args.eval_batch_size,
        episodes_per_app=args.episodes_per_app,
        log_interval_steps=args.log_interval_steps,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        trial=None,
    )

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = save_dir / args.checkpoint_name
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "embed_dim": args.embed_dim,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "d_ff": args.d_ff,
                "dropout": args.dropout,
                "attn_dropout": args.attn_dropout,
                "layer_scale_init": args.layer_scale_init,
                "width": args.width,
                "height": args.height,
                "num_buckets": args.num_buckets,
                "max_distance": args.max_distance,
                "log_base": args.log_base,
                "function_proj_hidden_dim": args.function_proj_hidden_dim,
                "vision_proj_hidden_dim": args.vision_proj_hidden_dim,
                "text_proj_hidden_dim": args.text_proj_hidden_dim,
                "proj_dropout": args.proj_dropout,
                "proj_init_scale": args.proj_init_scale,
            },
            "train_config": {
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "temperature": args.temperature,
                "contrastive_margin": args.contrastive_margin,
                "contrastive_margin_weight": args.contrastive_margin_weight,
                "grad_clip_norm": args.grad_clip_norm,
                "early_stopping_patience": args.early_stopping_patience,
                "train_classes_per_app": args.train_classes_per_app,
                "episodes_per_app": args.episodes_per_app,
                "seed": args.seed,
            },
            "best": {
                "epoch": result.best_epoch,
                "balanced_accuracy": result.best_score,
                "threshold": result.best_threshold,
                **result.best_metrics,
            },
            "split_mode": split_mode,
            "train_apps": sorted(train_split.keys()),
            "val_apps": sorted(val_split.keys()),
            "ood_apps": sorted(ood_split.keys()),
        },
        ckpt_path,
    )

    metrics_payload = {
        "best_epoch": result.best_epoch,
        "best_balanced_accuracy": result.best_score,
        "best_threshold": result.best_threshold,
        "val_metrics": result.best_metrics,
    }

    if ood_split:
        ood_metrics = evaluate_split_at_threshold(
            model=model,
            split_index=ood_split,
            feature_store=feature_store,
            device=device,
            eval_batch_size=args.eval_batch_size,
            threshold=result.best_threshold,
        )
        metrics_payload["ood_metrics"] = ood_metrics

    metrics_path = save_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved checkpoint: {ckpt_path}", flush=True)
    print(f"Saved metrics:    {metrics_path}", flush=True)


def run_optuna(args) -> None:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("optuna is required for --mode optuna. Install with: pip install optuna") from exc

    set_seed(args.seed)

    dataset_root = Path(args.dataset_root).resolve()
    train_apps = parse_csv_arg(args.train_apps)
    ood_apps = parse_csv_arg(args.ood_apps)
    train_split, val_split, ood_split, split_mode = build_dataset_splits(
        dataset_root=dataset_root,
        train_apps=train_apps,
        ood_apps=ood_apps,
        train_classes_per_app=args.train_classes_per_app,
        seed=args.seed,
    )

    print(
        f"[DATA] split_mode={split_mode}, train_apps={len(train_split)}, "
        f"val_apps={len(val_split)}, test_apps={len(ood_split)}",
        flush=True,
    )
    print("[Optuna] trial value metric: val_objective_score", flush=True)
    print("[Optuna] pruner: disabled", flush=True)

    validate_split_requirements(train_split, split_name="train")
    validate_split_requirements(val_split, split_name="val")
    feature_store = preload_feature_store([train_split, val_split])
    device = torch.device(args.device)

    if args.sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=args.seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=args.seed)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage if args.storage else None,
        load_if_exists=bool(args.storage),
        direction="maximize",
        sampler=sampler,
    )

    trial_log_dir = Path(args.save_dir).resolve() / "trial_logs"
    trial_log_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        model = None
        trial_log_path = trial_log_dir / f"trial_{trial.number:04d}.log"
        trial.set_user_attr("log_path", str(trial_log_path))
        trial.set_user_attr("epochs_completed", 0)
        print(f"[Trial {trial.number:03d}] log_file={trial_log_path}", flush=True)

        def _gpu_mem_snapshot(prefix: str) -> None:
            if not torch.cuda.is_available():
                return
            try:
                alloc_mb = torch.cuda.memory_allocated() / (1024**2)
                reserved_mb = torch.cuda.memory_reserved() / (1024**2)
                print(
                    f"[Trial {trial.number:03d}] {prefix} gpu_mem_alloc={alloc_mb:.1f}MB reserved={reserved_mb:.1f}MB",
                    flush=True,
                )
            except Exception:
                # Logging must never break training.
                pass

        with trial_log_capture(trial_log_path):
            print(f"[Trial {trial.number:03d}] detailed log started", flush=True)
            try:
                _gpu_mem_snapshot("start")
                model = build_model_from_args(args, trial=trial).to(device)

                if args.optuna_tune_optimizer:
                    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
                    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
                    temperature = trial.suggest_categorical("temperature", [0.03, 0.05, 0.07, 0.1])
                else:
                    lr = args.lr
                    weight_decay = args.weight_decay
                    temperature = args.temperature

                if args.optuna_tune_training_hparams:
                    contrastive_margin = trial.suggest_categorical("contrastive_margin", [0.02, 0.05, 0.1, 0.2])
                    contrastive_margin_weight = trial.suggest_categorical(
                        "contrastive_margin_weight",
                        [0.1, 0.25, 0.5, 1.0],
                    )
                    grad_clip_norm = trial.suggest_categorical("grad_clip_norm", [0.5, 1.0, 2.0, 5.0])
                else:
                    contrastive_margin = args.contrastive_margin
                    contrastive_margin_weight = args.contrastive_margin_weight
                    grad_clip_norm = args.grad_clip_norm

                print(
                    f"[Trial {trial.number:03d}] params={json.dumps(trial.params, ensure_ascii=False, sort_keys=True)}",
                    flush=True,
                )
                print(
                    "[Trial {n:03d}] effective_hparams="
                    "lr={lr:.8f} weight_decay={wd:.8f} temperature={temp:.4f} "
                    "contrastive_margin={cm:.4f} contrastive_margin_weight={cmw:.4f} "
                    "grad_clip_norm={gc:.4f}".format(
                        n=trial.number,
                        lr=float(lr),
                        wd=float(weight_decay),
                        temp=float(temperature),
                        cm=float(contrastive_margin),
                        cmw=float(contrastive_margin_weight),
                        gc=float(grad_clip_norm),
                    ),
                    flush=True,
                )

                result = train_one_run(
                    model=model,
                    train_split=train_split,
                    val_split=val_split,
                    feature_store=feature_store,
                    device=device,
                    epochs=args.epochs,
                    lr=lr,
                    weight_decay=weight_decay,
                    temperature=temperature,
                    contrastive_margin=contrastive_margin,
                    contrastive_margin_weight=contrastive_margin_weight,
                    grad_clip_norm=grad_clip_norm,
                    eval_batch_size=args.eval_batch_size,
                    episodes_per_app=args.episodes_per_app,
                    log_interval_steps=args.log_interval_steps,
                    early_stopping_patience=args.early_stopping_patience,
                    seed=args.seed + trial.number,
                    trial=trial,
                )

                trial.set_user_attr("best_epoch", result.best_epoch)
                trial.set_user_attr("best_threshold", result.best_threshold)
                trial.set_user_attr("best_macro_f1", result.best_metrics.get("macro_f1", 0.0))
                trial.set_user_attr("val_objective_score", result.optuna_objective_score)
                print(
                    "[Trial {n:03d}] finished val_objective={val_obj:.6f} best_epoch={ep} "
                    "val_best_bal_acc={best:.6f} train_calib_threshold={thr:.6f}".format(
                        n=trial.number,
                        val_obj=float(result.optuna_objective_score),
                        ep=int(result.best_epoch),
                        best=float(result.best_score),
                        thr=float(result.best_threshold),
                    ),
                    flush=True,
                )
                _gpu_mem_snapshot("end(success)")

                return float(result.optuna_objective_score)
            except optuna.TrialPruned as exc:
                reason = str(exc) if str(exc) else "no message"
                if not trial.user_attrs.get("prune_reason"):
                    trial.set_user_attr("prune_reason", "pruned")
                print(
                    f"[Trial {trial.number:03d}] PRUNED(reason): {reason}",
                    flush=True,
                )
                _gpu_mem_snapshot("end(pruned)")
                raise
            except RuntimeError as exc:
                msg = str(exc)
                lowered = msg.lower()
                cuda_error_signatures = (
                    "cuda error",
                    "cublas_status_not_initialized",
                    "out of memory",
                    "cudnn_status",
                )
                if any(sig in lowered for sig in cuda_error_signatures):
                    trial.set_user_attr("prune_reason", "cuda_runtime_failure")
                    print(
                        f"[Trial {trial.number:03d}] params(at-fail)={json.dumps(trial.params, ensure_ascii=False, sort_keys=True)}",
                        flush=True,
                    )
                    print(
                        f"[Trial {trial.number:03d}] CUDA runtime failure -> prune and continue: {msg}",
                        flush=True,
                    )
                    _gpu_mem_snapshot("end(cuda-fail)")
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    raise optuna.TrialPruned(f"CUDA runtime failure: {msg}")
                raise
            finally:
                if model is not None:
                    del model
                gc.collect()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    torch.cuda.empty_cache()
                    _gpu_mem_snapshot("post-cleanup")

    timeout = args.optuna_timeout_sec if args.optuna_timeout_sec > 0 else None
    search_started = time.time()

    def _should_count_budget(trial) -> bool:
        epochs_completed = int(trial.user_attrs.get("epochs_completed", 0) or 0)
        return epochs_completed >= 1

    counted_trials = sum(1 for t in study.trials if _should_count_budget(t))
    raw_attempts = len(study.trials)
    target_counted_trials = int(args.n_trials)
    max_raw_attempts = max(target_counted_trials * 5, target_counted_trials + 20)

    if counted_trials > 0:
        print(
            f"[Optuna] resume state: counted_trials={counted_trials}/{target_counted_trials} "
            f"raw_attempts={raw_attempts}",
            flush=True,
        )

    while counted_trials < target_counted_trials:
        if raw_attempts >= max_raw_attempts:
            print(
                f"[Optuna] reached max raw attempts ({max_raw_attempts}) before counted target "
                f"({counted_trials}/{target_counted_trials}). Stop search.",
                flush=True,
            )
            break

        remaining_timeout = None
        if timeout is not None:
            elapsed = time.time() - search_started
            remaining_timeout = max(0, int(timeout - elapsed))
            if remaining_timeout <= 0:
                print(
                    f"[Optuna] timeout reached before counted target "
                    f"({counted_trials}/{target_counted_trials}).",
                    flush=True,
                )
                break

        prev_len = len(study.trials)
        study.optimize(
            objective,
            n_trials=1,
            timeout=remaining_timeout,
            gc_after_trial=True,
        )
        if len(study.trials) <= prev_len:
            print(
                "[Optuna] no new trial executed (likely timeout). Stop search.",
                flush=True,
            )
            break

        last_trial = study.trials[-1]
        raw_attempts += 1
        epochs_completed = int(last_trial.user_attrs.get("epochs_completed", 0) or 0)
        counted = epochs_completed >= 1
        if counted:
            counted_trials += 1
        else:
            prune_reason = str(last_trial.user_attrs.get("prune_reason", "unknown"))
            print(
                f"[Optuna] trial {last_trial.number} NOT counted toward n_trials "
                f"(epochs_completed={epochs_completed}, reason={prune_reason})",
                flush=True,
            )

        print(
            f"[Optuna] progress counted={counted_trials}/{target_counted_trials} "
            f"raw_attempts={raw_attempts}",
            flush=True,
        )

    best_trial = study.best_trial
    print("[Optuna] best trial:", flush=True)
    print(f"  number={best_trial.number}", flush=True)
    print(f"  value={best_trial.value:.6f}", flush=True)
    print(f"  params={best_trial.params}", flush=True)

    out = {
        "best_trial_number": best_trial.number,
        "best_value": float(best_trial.value),
        "best_params": best_trial.params,
        "best_user_attrs": best_trial.user_attrs,
    }

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    best_json = save_dir / "optuna_best_trial.json"
    best_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved best trial: {best_json}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Screen-SBERT contrastive training with app-local sampling")

    parser.add_argument("--mode", choices=["train", "optuna"], default="train")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--train-apps", type=str, default="")
    parser.add_argument("--ood-apps", type=str, default="")
    parser.add_argument("--train-classes-per-app", type=int, default=16)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="e.g., cuda, cuda:0, cpu",
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--contrastive-margin", type=float, default=0.1)
    parser.add_argument("--contrastive-margin-weight", type=float, default=0.5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--episodes-per-app", type=int, default=0)
    parser.add_argument(
        "--log-interval-steps",
        type=int,
        default=60,
        help="Print running train metrics every N steps (<=0 disables step logging)",
    )
    parser.add_argument("--eval-batch-size", type=int, default=32)

    parser.add_argument("--save-dir", type=str, default="./runs/contrastive")
    parser.add_argument("--checkpoint-name", type=str, default="best.pt")
    parser.add_argument(
        "--training-defaults-checkpoint",
        type=str,
        default="",
        help="Load model/training defaults from an existing checkpoint",
    )

    # Manual model config (used in --mode train)
    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=1536)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attn-dropout", type=float, default=0.1)
    parser.add_argument("--layer-scale-init", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--num-buckets", type=int, default=32)
    parser.add_argument("--max-distance", type=int, default=128)
    parser.add_argument("--log-base", type=float, default=2.0)
    parser.add_argument("--function-proj-hidden-dim", type=int, default=192)
    parser.add_argument("--vision-proj-hidden-dim", type=int, default=192)
    parser.add_argument("--text-proj-hidden-dim", type=int, default=192)
    parser.add_argument("--proj-dropout", type=float, default=0.1)
    parser.add_argument("--proj-init-scale", type=float, default=0.1)

    # Optuna options
    parser.add_argument(
        "--n-trials",
        type=int,
        default=200,
        help="Target number of counted trials (only trials with >=1 completed epoch are counted)",
    )
    parser.add_argument("--study-name", type=str, default="screen_sbert_search")
    parser.add_argument("--storage", type=str, default="")
    parser.add_argument("--optuna-timeout-sec", type=int, default=0)
    parser.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Stop training early if val_bal_acc does not improve for this many consecutive epochs (<=0 disables)",
    )
    tune_optimizer_group = parser.add_mutually_exclusive_group()
    tune_optimizer_group.add_argument(
        "--optuna-tune-optimizer",
        dest="optuna_tune_optimizer",
        action="store_true",
        help="Tune lr/weight_decay/temperature in optuna mode (default: on)",
    )
    tune_optimizer_group.add_argument(
        "--no-optuna-tune-optimizer",
        dest="optuna_tune_optimizer",
        action="store_false",
        help="Disable optimizer hyperparameter tuning in optuna mode",
    )
    parser.set_defaults(optuna_tune_optimizer=True)

    tune_training_group = parser.add_mutually_exclusive_group()
    tune_training_group.add_argument(
        "--optuna-tune-training-hparams",
        dest="optuna_tune_training_hparams",
        action="store_true",
        help="Tune contrastive_margin/contrastive_margin_weight/grad_clip_norm in optuna mode (default: on)",
    )
    tune_training_group.add_argument(
        "--no-optuna-tune-training-hparams",
        dest="optuna_tune_training_hparams",
        action="store_false",
        help="Disable training hyperparameter tuning in optuna mode",
    )
    parser.set_defaults(optuna_tune_training_hparams=True)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    apply_training_defaults_from_checkpoint(args)

    if args.mode == "train":
        run_train(args)
        return 0

    if args.mode == "optuna":
        run_optuna(args)
        return 0

    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
