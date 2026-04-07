# Functional Semantics Embedding of GUI Screens for Knowledge-Augmented GUI Agents

This repository contains the model implementation code and dataset for the paper "**Functional Semantics Embedding of GUI Screens for Knowledge-Augmented GUI Agents**," submitted to **ACM MM 2026**.

---

In this model implementation code, the GUI Parsing Module is implemented based on Triton. 

The implementation is organized as:
- Triton model server (`gui_parser_server`)
- API proxy (`gui_parser_proxy`)

The proxy exposes APIs such as `merge_detect`, `parse_gui`, and `screen_sbert_embed`.

---

## 1) Environment Setup

### Python Virtual Environment

Create a Python 3.12 conda environment named `Screen-SBERT`, then install dependencies:

```bash
conda create -n Screen-SBERT python=3.12 -y
conda activate Screen-SBERT
pip install --upgrade pip
pip install -r requirements.txt
```

You may also use other virtual environments, such as venv.

### Docker

Docker must be installed in your environment.
If it is not installed, follow the steps below.

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

---

## 2) Download `server/models` from Hugging Face

Prebuilt model files are hosted at:

https://huggingface.co/user83kd9x/gui_parser_models

Download and restore the directory structure into this repository path (`server/models/...`):

```bash
hf download user83kd9x/gui_parser_models \
  --local-dir server/models
```

After download, you must remove `.cache` under `server/models`:

```bash
rm -rf server/models/.cache
```

Important: keep `--local-dir server/models` exactly as shown.
If you download to a different path, The Triton server container cannot correctly recognize the model path.

---

## 3) Build Docker Images

Build both server/proxy images:

```bash
bash build.sh
```

Expected images:

- `gui-parser_server:latest`
- `gui-parser_proxy:latest`

---

## 4) Build TensorRT Engines (`.plan`)

The GUI parsing server is designed to serve TensorRT-optimized engines.
Convert ONNX models into TensorRT plan files for your current GPU:

```bash
bash optimize.sh
```

Notes:

- This step can take significant time.
- You should see generated files like:
  - `server/models/*_engine/1/model.plan`

### CPU Fallback

- If optimization does not complete successfully, or if your environment cannot use a GPU, edit all `server/models/*_engine/config.pbtxt` files and set `platform: "onnxruntime_onnx"`.
- This makes the models load on CPU devices.
- Since GPU acceleration is not used in this mode, inference can take significantly longer.

---

## 5) Run Server/Proxy Containers

Start both containers:

```bash
bash run.sh
```

### 5.1 Check Proxy Logs

```bash
docker logs gui_parser_proxy
```

You should see Uvicorn startup logs similar to:

```text
INFO:     Uvicorn running on http://0.0.0.0:4023 (Press CTRL+C to quit)
INFO:     Started parent process [1]
INFO:     Started server process [8]
INFO:     Started server process [10]
INFO:     Waiting for application startup.
INFO:     Waiting for application startup.
INFO:     Started server process [9]
INFO:     Application startup complete.
INFO:     Application startup complete.
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Started server process [11]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 5.2 Check Triton Logs

```bash
docker logs gui_parser_server
```

Look for:

```text
... Started GRPCInferenceService at 0.0.0.0:4001
... Started HTTPService at 0.0.0.0:4000
... Started Metrics Service at 0.0.0.0:4002
```

Before these three lines appear, Triton is still loading models.
When they appear, model loading is complete and serving is ready.

### 5.3 Stop Containers

```bash
bash stop.sh
```

---

## 6) Run Examples

### 6.1 GUI/Text Detection + Merge (`merge_detect`)

```bash
python example/merge_detect_example.py example/example_coupang.png
```

Outputs are saved under `example/results/`:

- merged detection JSON
- input image with drawn bounding boxes

### 6.2 GUI Parsing Output for Screen-SBERT Input (`parse_gui`)

```bash
python example/parse_gui_example.py example/example_coupang.png
```

Outputs are saved under `example/results/` as four `.npy` files:

- `bbox.npy`
- `function_embedding.npy`
- `text_embedding.npy`
- `vision_embedding.npy`

---

## Quick Start (Command Order)

```bash
hf download user83kd9x/gui_parser_models --local-dir server/models
rm -rf server/models/.cache
bash build.sh
bash optimize.sh
bash run.sh
python example/merge_detect_example.py example/example_coupang.png
python example/parse_gui_example.py example/example_coupang.png
```

---

## 7) Screen-SBERT Model Training and Evaluation

To run model training and evaluation under `Screen-SBERT/`, datasets must be pre-parsed in advance using this GUI Parsing Module Triton server/proxy implementation.

For details on Screen-SBERT training and evaluation, please refer to `Screen-SBERT/README.md`.

---

## 8) End-to-End Screen-SBERT Embedding API Example

We converted the checkpoint used in our experiments to ONNX format and included it in the Triton model server.
You can test the end-to-end API that takes a screenshot image and returns one functional semantics embedding vector.

```bash
python example/screen_sbert_embed_example.py example/example_coupang.png
```

Output:

- `example/results/<image_stem>_screen_sbert_embedding.npy`
- printed embedding shape in terminal

---

## Screenshot Embedding using MLLM

This repository also includes a prompt for embedding screenshots using an MLLM.

The procedure is as follows:

1. Provide the prompt in `gpt_embedding_prompt.py` to the MLLM along with the screenshot image.
2. Extract the text from the "Functions" section of the response.
3. Embed the text using the GUI Parser server and proxy provided in this repository.

Text embeddings can be used as follows.

```bash
python example/text_embedding_example.py "Hello, World!"
```

---

## Other Repositories Related to the Paper

### Benchmark Dataset Used for Action Decision Experiments in This Study
https://huggingface.co/datasets/user83kd9x/knowledge_agent_benchmark

### Embedding Results for All Screenshots in the Benchmark Using Screen-SBERT, Along with Code for Clustering and Knowledge Merging
https://huggingface.co/datasets/user83kd9x/screenshot_clustering
