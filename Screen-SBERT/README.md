# Screen-SBERT

Screen-SBERT is a contrastive-learning model for screen-level embedding from pre-parsed GUI element features.

## What This Project Expects

This project **does not train directly from raw screenshots**.
It trains from precomputed GUI parser outputs per screenshot:
Please pre-parse screenshots in advance through the GUI Parsing Module Triton server and proxy.

- `bbox`
- `function_embedding`
- `text_embedding`
- `vision_embedding`

Each screenshot may have a different number of GUI elements (`N`), and variable-length inputs are supported.

Supported sample storage (inside each `<class_name>/`):

- **Sample directory**: one folder per sample containing  
  `bbox.npy`, `function_embedding.npy`, `text_embedding.npy`, `vision_embedding.npy`
- **NPZ sample file**: one `.npz` file containing those same four keys
- `parse_gui.npz` is also supported (including nested paths under the class directory)

You can mix these formats in the same class folder.

### Dataset Used in Experiments

To download the dataset used in the experiments of this study, please follow the procedure below.

```bash
hf download user83kd9x/screen_sbert_dataset \
  --repo-type dataset \
  --local-dir dataset
```
```bash
unzip dataset/gui_parsing.zip -d dataset/gui_parsing
unzip dataset/screenshots.zip -d dataset/screenshots
```

This dataset inherently satisfies the structural requirements described below.

---

## 1) Dataset Structure Requirements

### Train/Validation Split Organization

This section describes **how train/val/test splits are organized on disk** (not the sample tensor format itself).

You can choose one of the following split layouts:

### A. Predefined split directories (recommended)

```text
<dataset_root>/
  train/
    <app_name>/
      <class_name>/
        <sample_id>/
          bbox.npy
          function_embedding.npy
          text_embedding.npy
          vision_embedding.npy
        ...
  val/ or validation/
    <app_name>/
      <class_name>/
        <sample_id>/
          bbox.npy
          function_embedding.npy
          text_embedding.npy
          vision_embedding.npy
  test/                    # optional for training; useful for OOD metrics
    <app_name>/
      <class_name>/
        <sample_id>/
          bbox.npy
          function_embedding.npy
          text_embedding.npy
          vision_embedding.npy
```

### B. Single-root dataset (automatic random class split)

If `train/` and `val/` folders do not exist, the script treats `<dataset_root>/<app>/<class>/...` as a single pool and performs class split per app.

```text
<dataset_root>/
  <app_name>/
    <class_name>/
      <sample_id>/
        bbox.npy
        function_embedding.npy
        text_embedding.npy
        vision_embedding.npy
```

### Per-sample tensor constraints

For every sample:

- `bbox`: shape `[N, 4]`
- `function_embedding`: shape `[N, 1024]`
- `text_embedding`: shape `[N, 1024]`
- `vision_embedding`: shape `[N, 768]`

And all four tensors must share the same `N`.

### Minimum data constraints (important)

For both train and validation:

- each app must have at least **2 classes**
- each class must have at least **2 samples**

Otherwise training will fail.

---

## 2) Find Best Hyperparameters with Optuna

You have two ways:

- **Direct Optuna search only**: `train_contrastive.py --mode optuna`
- **Pipeline (recommended)**: run Optuna search first, then automatically launch final training

### 2-1) Direct Optuna search

Run from repository root:

```bash
python Screen-SBERT/train_contrastive.py \
  --mode optuna \
  --dataset-root /absolute/path/to/dataset/gui_parsing \
  --save-dir /absolute/path/to/output/optuna \
  --epochs 30 \
  --n-trials 200 \
  --train-classes-per-app 16 \
  --device cuda
```

Key options in this command:

- `--mode optuna`: run hyperparameter search mode.
- `--dataset-root`: input dataset root path.
- `--save-dir`: directory where Optuna artifacts are saved.
- `--epochs`: max epochs per trial.
- `--n-trials`: target number of counted trials.
- `--train-classes-per-app`: number of classes per app used for train split when random class split is used.
- `--device`: training device (`cuda`, `cuda:0`, or `cpu`).

What it does:

- samples model hyperparameters from `Screen-SBERT/optuna_train_pipeline.py`
- optionally tunes optimizer/training hparams (`lr`, `weight_decay`, `temperature`, `contrastive_margin`, etc.)
- writes one log file per trial under `trial_logs/`
- saves best trial summary to `optuna_best_trial.json`

Current Optuna trial value is validation-based (`val_objective_score`).

### 2-2) Recommended one-command pipeline (Optuna + final training)

```bash
python Screen-SBERT/optuna_train_pipeline.py \
  --dataset-root /absolute/path/to/dataset/gui_parsing \
  --output-root /absolute/path/to/output/optuna_pipeline \
  --n-trials 200 \
  --optuna-epochs 30 \
  --final-epochs 30 \
  --device cuda
```

Key options in this command:

- `--dataset-root`: input dataset root path.
- `--output-root`: base output directory for this pipeline run.
- `--n-trials`: target number of Optuna counted trials.
- `--optuna-epochs`: max epochs per Optuna trial.
- `--final-epochs`: max epochs for final training stage.
- `--device`: device used by both Optuna and final training.

Outputs:

- `<run>/optuna/optuna_best_trial.json`
- `<run>/optuna/trial_logs/trial_XXXX.log`
- `<run>/final_train/final_best.pt`
- `<run>/final_train/metrics.json`
- `<run>/pipeline_summary.json`

---

## 3) Run Final Training

If you already decided your training settings and only want final training:

```bash
python Screen-SBERT/train_contrastive.py \
  --mode train \
  --dataset-root /absolute/path/to/dataset/gui_parsing \
  --save-dir /absolute/path/to/output/final_train \
  --checkpoint-name final_best.pt \
  --epochs 30 \
  --device cuda
```

Key options in this command:

- `--mode train`: run regular training mode.
- `--dataset-root`: input dataset root path.
- `--save-dir`: output directory for checkpoint/metrics.
- `--checkpoint-name`: output checkpoint filename.
- `--epochs`: max training epochs.
- `--device`: training device.

You can also initialize model/training defaults from an existing checkpoint:

```bash
python Screen-SBERT/train_contrastive.py \
  --mode train \
  --dataset-root /absolute/path/to/dataset/gui_parsing \
  --save-dir /absolute/path/to/output/final_train \
  --training-defaults-checkpoint /absolute/path/to/existing_checkpoint.pt \
  --device cuda
```

Key option in this variant:

- `--training-defaults-checkpoint`: load model/training defaults from an existing checkpoint's config block.

---

## 4) Export to ONNX (for Triton Server deployment)

### Why ONNX here?

The main reason to export ONNX is to make deployment to **NVIDIA Triton Inference Server** straightforward.
From ONNX, you can build a TensorRT plan and serve the model reliably for downstream tests and production-like inference.

### Export command

```bash
python Screen-SBERT/export_onnx.py \
  --checkpoint /absolute/path/to/final_best.pt \
  --onnx-out /absolute/path/to/screen_sbert.onnx \
  --device cpu \
  --batch-size 1 \
  --num-gui 64 \
  --opset 17
```

Key options in this command:

- `--checkpoint`: trained checkpoint file (`.pt`/`.pth`).
- `--onnx-out`: output ONNX file path.
- `--device`: export device (`cpu` or `cuda`).
- `--batch-size`: dummy batch size used during tracing.
- `--num-gui`: dummy GUI token length used during tracing.
- `--opset`: ONNX opset version.

Optional flags:

- `--fp16-dummy`: uses FP16 dummy tensors during export
- `--no-dynamic-axes`: disables dynamic axes

The script prints an example `trtexec` command to create a TensorRT plan (`.plan`) from the ONNX model.

For full CLI options, run:

- `python Screen-SBERT/train_contrastive.py --help`
- `python Screen-SBERT/optuna_train_pipeline.py --help`
- `python Screen-SBERT/export_onnx.py --help`

---

## Core Files (Current)

- `train_contrastive.py`: core training script and Optuna trial execution logic
- `optuna_train_pipeline.py`: end-to-end pipeline runner (Optuna search -> final training) and model search-space definition
- `export_onnx.py`: checkpoint-to-ONNX exporter (includes ONNX/TensorRT helper utilities)
- `models/`: model architecture
  - `embedding.py`
  - `encoder.py`
  - `model.py`

---

## Environment Notes

- Python 3.10+ recommended
- Optuna is required for `--mode optuna`
- use GPU (`--device cuda`) for practical training speed

---

## Troubleshooting

### "must contain at least 2 classes" / "must contain at least 2 samples"

Your split has too few classes or too few samples per class. Fix dataset composition first.

### Tensor shape errors (e.g., expected `[N,1024]`)

Your parsed feature files are malformed or from a different embedding model version. Re-generate features with the expected dimensions.

### CUDA runtime failures during Optuna

Some sampled configs can be unstable on specific hardware/driver combinations. The script prunes such trials and continues.

---

## Typical Workflow Summary

1. Prepare parsed GUI feature dataset with required structure and shapes.
2. Run `optuna_train_pipeline.py` to get best config and final checkpoint.
3. Export final checkpoint to ONNX via `export_onnx.py`.
4. Build TensorRT plan from ONNX and deploy on Triton for downstream tests.
