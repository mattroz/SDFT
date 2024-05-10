accelerate launch train_ti_sdxl.py \
  --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 \
  --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
  --allow_tf32 \
  --mixed_precision="fp16" \
  --train_data_dir=datasets/skull \
  --learnable_property="style" \
  --placeholder_token="<skull_lamp>" \
  --initializer_token="skull" \
  --num_vectors=8 \
  --resolution=1024 \
  --repeats=1 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 --gradient_checkpointing \
  --max_train_steps=5000 \
  --learning_rate=3e-3 \
  --lr_scheduler="piecewise_constant" \
  --lr_warmup_steps=30 \
  --output_dir="runs/ti_run" \
  --validation_prompt="A painting of Eiffel tower in the style of <skull_lamp>" \
  --num_validation_images=4 \
  --validation_steps=100 \
  --embeddings_save_steps=500 \
  --save_images_on_disk \
  --use_random_flip \
  --use_center_crop \
  --seed=1337 \