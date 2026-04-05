export HF_HOME=/data/
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=1

python run_single.py \
  --model_id <Wan-AI/Wan2.1-T2V-1.3B-Diffusers/black-forest-labs/FLUX.1-dev> \
  --backend wan \
  --mixed_precision fp16 \
  --resume_path ckpts/<backend_degradation.pt> \
  --t0 0.4 \
  --t_min 0.0001 \
  --bridge_sampler_steps 20 \
  --input_type <video/image> \
  --single_input <path_to_sample>