# Your Pre-trained Diffusion Model Secretly Knows Restoration

Official code for the paper **"Your Pre-trained Diffusion Model Secretly Knows Restoration"** 

[Sudarshan Rajagopalan](https://sudraj2002.github.io/) | [Vishal M. Patel](https://scholar.google.com/citations?user=AkEXTbIAAAAJ&hl=en)

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://sudraj2002.github.io/yptpage/) [![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/)

## Overview

## Setup

### 1. Creating environment

```bash
conda create -n ypt python=3.10 -y
conda activate ypt
pip install --no-build-isolation -r requirements.txt
pip install --no-build-isolation flash-attn==2.8.3
pip install --no-build-isolation pyiqa
python -m pip install "setuptools==80.9.0"
```

### 2. Hugging Face setup

This code downloads the backbone models from Hugging Face on first use. Make sure you:

- have a working Hugging Face account
- have access to the required model repositories

Optional environment variables used in the provided shell scripts:

```bash
export HF_HOME=/path/to/hf_cache
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
```

## Data download

This repository includes metadata as JSON files for the evaluation sets under ```test_img_jsons/```, and 
```test_vid_jsons```. Download the evaluation data for images from [here]() and videos from [here](). Extract the 
files to ```test_img_data/``` and ```test_vid_data```,  respectively.

### JSON format for images

Each entry should look like:

```json
[
  {
    "image_path": "test_img_data/POLED/input/input_0000.png",
    "target_path": "test_img_data/POLED/gt/gt_0000.png",
    "degradation": "LowLight",
    "dataset": "POLED"
  }
]
```

### JSON format for videos

Each entry should look like:

```json
[
  {
    "image_path": [
      "test_vid_data/SDSD/input/pair37/0082.png",
      "test_vid_data/SDSD/input/pair37/0083.png"
    ],
    "target_path": [
      "test_vid_data/SDSD/GT/pair37/0082.png",
      "test_vid_data/SDSD/GT/pair37/0083.png"
    ],
    "degradation": "Low-light",
    "dataset": "SDSD"
  }
]
```

## Running Evaluation

### FLUX image evaluation

Use the provided script:

```bash
bash inference_flux.sh
```

Or run a single dataset manually:

```bash
python inference.py \
  --model_id black-forest-labs/FLUX.1-dev \
  --backend flux \
  --mixed_precision fp16 \
  --resume_path ckpts/flux_haze.pt \
  --json_path test_img_jsons/RESIDE.json \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20
```

### WAN video evaluation

Use the provided script:

```bash
bash inference_wan.sh
```

Or run a single dataset manually:

```bash
python inference.py \
  --model_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --backend wan \
  --mixed_precision fp16 \
  --resume_path ckpts/wan_lowlight.pt \
  --json_path test_vid_jsons/SDSD_test.json \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20
```

To calculate DOVER, setup the DOVER codebase and environments according to the official [codebase](). After setup, 
place ```calc_dover.py``` in the DOVER codebase. To generate the videos from the WAN result directory, run 
```bash
python frames_to_vid.py \
  --root <results_dir> \
  --out_root <output_directory> \
  --fps 15 \
  --methods wan \
  --min_frames 1 \
  --size 512 \
  --lossless
```

Subsequently, in the DOVER environment run
```bash
python calc_dover.py \
  --root_dir <output_directory_from_previous> \
  --methods wan
```
Finally, inspect the generated ```dover_summary.csv``` for the values.

### Mixed-degradation inference

To combine multiple prompt checkpoints at inference time:

```bash
python inference.py \
  --model_id black-forest-labs/FLUX.1-dev \
  --backend flux \
  --mixed_precision fp16 \
  --mixed_paths ckpts/flux_lowlight.pt ckpts/flux_blur.pt \
  --json_path test_img_jsons/LOLBlur.json \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20
```

## Single-Sample Inference

Use `run_single.py` when you want to restore one image or one video.

Template:

```bash
python run_single.py \
  --model_id <model_id> \
  --backend <wan|flux> \
  --mixed_precision fp16 \
  --resume_path ckpts/<backend_degradation.pt> \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20 \
  --input_type <image|video> \
  --single_input <path_to_sample>
```

### Example: single image with FLUX

```bash
python run_single.py \
  --model_id black-forest-labs/FLUX.1-dev \
  --backend flux \
  --mixed_precision fp16 \
  --resume_path ckpts/flux_rain.pt \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20 \
  --input_type image \
  --single_input /path/to/image.png
```

### Example: single video with WAN

```bash
python run_single.py \
  --model_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --backend wan \
  --mixed_precision fp16 \
  --resume_path ckpts/wan_snow.pt \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20 \
  --input_type video \
  --single_input /path/to/video.mp4
```

For video single-sample inference, `--single_input` can be either:

- a video file such as `.mp4`, `.mov`, `.avi`, `.mkv`, or `.webm`
- a directory containing ordered frame images

## Gradio Demo

Launch the demo with:

```bash
bash run_gradio.sh
```

or:

```bash
python app.py
```

The demo:

- uses `wan` for video input
- uses `flux` for image input
- lets you select one degradation checkpoint or multiple checkpoints for mixed inference
- saves temporary outputs under `gradio_tmp/`

## Outputs

### Evaluation

Results are written under:

```bash
results/<backend>/<dataset_name>/
```

### Single-sample inference

Outputs are written under:

```bash
results/<backend>/single/pred/
```

For WAN video inference, the script saves:

- individual restored frames
- `result.mp4`

## Citation

If you find this work useful, please cite:

```bibtex

```
