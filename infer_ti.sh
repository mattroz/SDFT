accelerate launch run_ti_inference.py \
  --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 \
  --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
  --output_dir=runs/ti_run_style \
  --path_to_embeddings=runs/ti_run/ti-embeddings-final.safetensors \
  --resolution=1024 \
  --num_images_to_generate=1 \
  --guidance_scale=5.0 \
  --num_inference_steps=50 \
  --placeholder_token="<skull_lamp>" \
  --prompt="A <skull_lamp>, made of lego" \
  --negative_prompt="logo, watermark, text, blurry, bad quality" \
  --seed=1337