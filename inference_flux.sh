export HF_HOME=/data/
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

python inference.py \
  --model_id black-forest-labs/FLUX.1-dev \
  --backend flux \
  --mixed_precision fp16 \
  --mixed_paths ckpts/flux_blur.pt ckpts/flux_lowlight.pt \
  --json_path test_img_jsons/POLED.json \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20

python inference.py \
  --model_id black-forest-labs/FLUX.1-dev \
  --backend flux \
  --mixed_precision fp16 \
  --mixed_paths ckpts/flux_lowlight.pt ckpts/flux_blur.pt \
  --json_path test_img_jsons/TOLED.json \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20

python inference.py \
  --model_id black-forest-labs/FLUX.1-dev \
  --backend flux \
  --mixed_precision fp16 \
  --resume_path ckpts/flux_blur.pt \
  --json_path test_img_jsons/4KRD.json \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20 \

python inference.py \
  --model_id black-forest-labs/FLUX.1-dev \
  --backend flux \
  --mixed_precision fp16 \
  --resume_path ckpts/flux_snow.pt \
  --json_path test_img_jsons/WeatherBenchSnow.json \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20

python inference.py \
  --model_id black-forest-labs/FLUX.1-dev \
  --backend flux \
  --mixed_precision fp16 \
  --resume_path ckpts/flux_rain.pt \
  --json_path test_img_jsons/LHP.json \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20

python inference.py \
  --model_id black-forest-labs/FLUX.1-dev \
  --backend flux \
  --mixed_precision fp16 \
  --resume_path ckpts/flux_lowlight.pt \
  --json_path test_img_jsons/SICE.json \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20

python inference.py \
  --model_id black-forest-labs/FLUX.1-dev \
  --backend flux \
  --mixed_precision fp16 \
  --resume_path ckpts/flux_haze.pt \
  --json_path test_img_jsons/HazeRD.json \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20
  
python inference.py \
  --model_id black-forest-labs/FLUX.1-dev \
  --backend flux \
  --mixed_precision fp16 \
  --mixed_paths ckpts/flux_snow.pt ckpts/flux_haze.pt \
  --json_path test_img_jsons/CDD_haze_snow.json \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20
  
python inference.py \
  --model_id black-forest-labs/FLUX.1-dev \
  --backend flux \
  --mixed_precision fp16 \
  --mixed_paths ckpts/flux_lowlight.pt ckpts/flux_blur.pt \
  --json_path test_img_jsons/LOLBlur.json \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20

python inference.py \
  --model_id black-forest-labs/FLUX.1-dev \
  --backend flux \
  --mixed_precision fp16 \
  --resume_path ckpts/flux_haze.pt \
  --json_path test_img_jsons/RESIDE.json \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20

python inference.py \
  --model_id black-forest-labs/FLUX.1-dev \
  --backend flux \
  --mixed_precision fp16 \
  --resume_path ckpts/flux_snow.pt \
  --json_path test_img_jsons/Snow100k.json \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20

python inference.py \
  --model_id black-forest-labs/FLUX.1-dev \
  --backend flux \
  --mixed_precision fp16 \
  --resume_path ckpts/flux_rain.pt \
  --json_path test_img_jsons/Rain100H.json \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20

python inference.py \
  --model_id black-forest-labs/FLUX.1-dev \
  --backend flux \
  --mixed_precision fp16 \
  --resume_path ckpts/flux_rain.pt \
  --json_path test_img_jsons/Rain100L.json \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20

python inference.py \
  --model_id black-forest-labs/FLUX.1-dev \
  --backend flux \
  --mixed_precision fp16 \
  --resume_path ckpts/flux_blur.pt \
  --json_path test_img_jsons/GoPro.json \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20

python inference.py \
  --model_id black-forest-labs/FLUX.1-dev \
  --backend flux \
  --mixed_precision fp16 \
  --resume_path ckpts/flux_lowlight.pt \
  --json_path test_img_jsons/LOLv1.json \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20
