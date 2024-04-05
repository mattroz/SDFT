accelerate launch run_lora_inference.py \
  --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 \
  --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
  --output_dir=runs/lora_v1/ \
  --lora_checkpoint_path=runs/lora_v1/checkpoint-1100/ \
  --resolution=1024 \
  --num_images_to_generate=1 \
  --guidance_scale=5.0 \
  --num_inference_steps=40 \
  --prompt="a picture of a supermassive black hole devouring the galaxy, far away view, cinematic picture" \
  --negative_prompt="logo, watermark, text, blurry" \
  --seed=1337

  # candid photo 
  #extra sword, crippled sword, extra fingers, prolapsed, extra legs, extra arms, fused fingers, fused legs, bad anatomy, bad proportions, crippled face, crippled limbs