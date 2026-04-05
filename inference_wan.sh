export HF_HOME=/data/
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=1

python inference.py \
  --model_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --backend wan \
  --mixed_precision fp16 \
  --resume_path ckpts/wan_lowlight.pt \
  --json_path test_vid_jsons/loliphone_test_pruned.json \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20

python inference.py \
  --model_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --backend wan \
  --mixed_precision fp16 \
  --resume_path ckpts/wan_lowlight.pt \
  --json_path test_vid_jsons/SDSD_test.json \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20

python inference.py \
  --model_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --backend wan \
  --mixed_precision fp16 \
  --resume_path ckpts/wan_blur.pt \
  --json_path test_vid_jsons/GoPro_test.json \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20


python inference.py \
  --model_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --backend wan \
  --mixed_precision fp16 \
  --resume_path ckpts/wan_blur.pt \
  --json_path test_vid_jsons/4KRD_test.json \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20


python inference.py \
  --model_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --backend wan \
  --mixed_precision fp16 \
  --resume_path ckpts/wan_snow.pt \
  --json_path test_vid_jsons/RVSD_test.json \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20


python inference.py \
  --model_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --backend wan \
  --mixed_precision fp16 \
  --resume_path ckpts/wan_snow.pt \
  --json_path test_vid_jsons/aau.json \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20

python inference.py \
  --model_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --backend wan \
  --mixed_precision fp16 \
  --resume_path ckpts/wan_rain.pt \
  --json_path test_vid_jsons/SPAC_test_real.json \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20

python inference.py \
  --model_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --backend wan \
  --mixed_precision fp16 \
  --resume_path ckpts/wan_rain.pt \
  --json_path test_vid_jsons/LasVR_test.json \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20 \

python inference.py \
  --model_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --backend wan \
  --mixed_precision fp16 \
  --resume_path ckpts/wan_haze.pt \
  --json_path test_vid_jsons/RHVD_test_pruned.json \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20

python inference.py \
  --model_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --backend wan \
  --mixed_precision fp16 \
  --resume_path ckpts/wan_haze.pt \
  --json_path test_vid_jsons/REVIDE_test.json \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20