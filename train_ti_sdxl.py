# Adapted from HuggingFace diffusers scripts
# https://github.com/huggingface/diffusers

import math
import shutil

import datasets
import json
import numpy as np
import torch
import pathlib
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers

from functools import partial

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed

from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel, CLIPTextModelWithProjection

import safetensors
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import convert_state_dict_to_diffusers
 
from src.utils.hf_helpers import unwrap_model, get_trainable_parameters_str
from src.trackers import FileSystemTracker
from src.data.dataset import TextualInversionDataset, collate_fn
from src.methods.textual_inversion.arguments import parse_train_args
from src.utils.logger import get_logger


def get_checkpoints_directories(output_dir):
    dirs = list(file.name for file in output_dir.iterdir() if file.is_dir())
    dirs = [d for d in dirs if str(d).startswith("checkpoint")]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    return dirs


def save_embeddings(text_encoder_one, text_encoder_two, placeholder_token_ids_one, placeholder_token_ids_two, save_path, safe_serialization=True):
    logger.info("Saving embeddings")
    
    embeddings_one = (
        text_encoder_one
        .text_model.embeddings.token_embedding
        .weight[placeholder_token_ids_one]
    )
    embeddings_two = (
        text_encoder_two
        .text_model.embeddings.token_embedding
        .weight[placeholder_token_ids_two]
    )

    embeddings_dict = {
        "text_encoder_one": embeddings_one.detach().cpu(),
        "text_encoder_two": embeddings_two.detach().cpu(),
    }

    if safe_serialization:
        safetensors.torch.save_file(embeddings_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(embeddings_dict, save_path)


def get_token_embeddings(text_encoder):
    token_embeddings = text_encoder.text_model.embeddings.token_embedding.weight.data.clone()
    return token_embeddings


def update_placeholder_tokens_only(text_encoder, tokenizer, placeholder_token_ids, original_token_embeddings):
    indices_without_update = torch.ones(len(tokenizer), dtype=torch.bool)
    indices_without_update[placeholder_token_ids] = False
    text_encoder.text_model.embeddings.token_embedding.weight.data[indices_without_update] = \
        original_token_embeddings[indices_without_update]


def inject_placeholder_tokens(tokenizer, text_encoder, placeholder_token, initializer_token, num_vectors):
    placeholder_tokens_str = [placeholder_token]

    for i in range(1, num_vectors):
        placeholder_tokens_str.append(f"{placeholder_token}_{i}")

    num_tokens_added = tokenizer.add_tokens(placeholder_tokens_str)
    logger.info(f"Added {num_tokens_added} tokens to the tokenizer.")
    if num_tokens_added != num_vectors:
        raise ValueError(
            f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    # Now get initializer_token id
    token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens_str)

    # As we added new tokens - resize Embedding layer
    text_encoder.resize_token_embeddings(len(tokenizer))
    resized_token_embeddings = text_encoder.text_model.embeddings.token_embedding.weight.data
    logger.info(f"Resized token embeddings shape of {type(text_encoder).__name__}: {resized_token_embeddings.shape}.")

    # Substitute placeholder tokens with the initializer token
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            resized_token_embeddings[token_id] = resized_token_embeddings[initializer_token_id].clone()

    return placeholder_token_ids


def validate(args, accelerator, unet, vae, text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two, epoch, weight_dtype):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    # create pipeline
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        unet=unet,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    pipeline_args = {"prompt": args.validation_prompt, "negative_prompt": "blurry, bad quality, distorted"}

    with torch.cuda.amp.autocast():
        images = [
            pipeline(**pipeline_args, generator=generator).images[0]
            for _ in range(args.num_validation_images)
        ]
    
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        elif tracker.name == "file_system_tracker":
            if args.save_images_on_disk:
                tracker.save_images(images, epoch)
        else:    
            raise NotImplementedError("Only tensorboard and file_system_tracker are supported for validation logging.")

    del pipeline
    torch.cuda.empty_cache()


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    # text_encoders = [CLIPTextModel, CLIPTextModelWithProjection]
    # In paper: CLIP ViT-L & OpenCLIP ViT-bigG
    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            text_input_ids = tokenizers[i](
                prompt,
                padding="max_length",
                max_length=tokenizers[i].model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder,
        # i.e. take features from the EOS token. Example is here:
        # https://github.com/mattroz/miniCLIP/blob/main/src/model/clip.py#L58

        # Paper uses pooled embeddings from OpenCLIP ViT-bigG model (Table 1 in the paper)
        pooled_prompt_embeds = prompt_embeds[0]  # overwrites by the second text_encoder, which pooled outputs has [bs, 1280] size (SDXL paper, Table 1)
        prompt_embeds = prompt_embeds[-1][-2]   # CLIPTextModel: [bs, 77, 768] | CLIPTextModelWithProjection: [bs, 77, 1280]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1) # [bs, 77, 2048]
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1) # [bs, 1280]
    return prompt_embeds, pooled_prompt_embeds


def main(args):
    logging_dir = pathlib.Path(args.output_dir, args.logging_dir)

    filesystem_tracker = FileSystemTracker(logging_dir=logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=[args.report_to, filesystem_tracker],
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Save placeholder token name to logging dir
    filesystem_tracker.log_to_file(
        {
            "placeholder_token": args.placeholder_token, 
            "initializer_token": args.initializer_token
        }, 
        "textual_inversion_params.json")

    unwrap_model_ = partial(unwrap_model, accelerator)

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # For mixed precision training we cast all non-trainable weights to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    # CLIP ViT-L in the paper
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="text_encoder", 
        revision=args.revision, 
        variant=args.variant
    )
    # OpenCLIP ViT-bigG in the paper
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="text_encoder_2", 
        revision=args.revision, 
        variant=args.variant
    )
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    # Gather all models' configs to log them
    if accelerator.is_main_process:
        path_to_save_configs = pathlib.Path(logging_dir, "model_configs")
        path_to_save_configs.mkdir(parents=True, exist_ok=True)        
        models_to_log = {
            "noise_scheduler": noise_scheduler,
            "text_encoder_one": text_encoder_one,
            "text_encoder_two": text_encoder_two,
            "vae": vae,
            "unet": unet,
        }
        for model_name in models_to_log:
            path_to_save_config = pathlib.Path(path_to_save_configs, f"{model_name}.json")
            config = models_to_log[model_name].config
            config = config.to_dict() if isinstance(config, PretrainedConfig) else config
            with open(path_to_save_config, "w") as f:
                json.dump(config, f, indent=4)

    # CONCEPT TOKENS INJECTION
    placeholder_token_ids_one = inject_placeholder_tokens(tokenizer_one, 
                                                          text_encoder_one, 
                                                          args.placeholder_token, 
                                                          args.initializer_token, 
                                                          args.num_vectors)
    placeholder_token_ids_two = inject_placeholder_tokens(tokenizer_two, 
                                                          text_encoder_two, 
                                                          args.placeholder_token, 
                                                          args.initializer_token, 
                                                          args.num_vectors)
    # CONCEPT TOKENS INJECTION END

    # We only train the token embeddings from text_encoder_one and text_encoder_two
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder_one.text_model.embeddings.token_embedding.requires_grad_(True)
    text_encoder_two.text_model.embeddings.token_embedding.requires_grad_(True)

    # Move unet, vae and text_encoders to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    unet.to(accelerator.device, dtype=weight_dtype)

    if args.pretrained_vae_model_name_or_path is None:
        vae.to(accelerator.device, dtype=torch.float32)
    else:
        vae.to(accelerator.device, dtype=weight_dtype)

    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    logger.info(f"{type(text_encoder_one).__name__} TI: {get_trainable_parameters_str(text_encoder_one)}", main_process_only=True)
    logger.info(f"{type(text_encoder_two).__name__} TI: {get_trainable_parameters_str(text_encoder_two)}", main_process_only=True)

    if args.gradient_checkpointing:
        text_encoder_one.gradient_checkpointing_enable()
        text_encoder_two.gradient_checkpointing_enable()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        cast_training_params([text_encoder_one, text_encoder_two], dtype=torch.float32)

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimizer = (
        list(text_encoder_one.text_model.embeddings.token_embedding.parameters()) + 
        list(text_encoder_two.text_model.embeddings.token_embedding.parameters())
    )
    optimizer = optimizer_class(
        params_to_optimizer,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    placeholder_token_one = " ".join(tokenizer_one.convert_ids_to_tokens(placeholder_token_ids_one))
    placeholder_token_two = " ".join(tokenizer_two.convert_ids_to_tokens(placeholder_token_ids_two))
    assert placeholder_token_one == placeholder_token_two, \
        f"Decoded placeholder tokens both tokenizers must be equal: {placeholder_token_one} != {placeholder_token_two}"

    train_dataset = TextualInversionDataset(
        data_root=args.train_data_dir,
        tokenizer_one=tokenizer_one,
        tokenizer_two=tokenizer_two,
        resolution=args.resolution,
        placeholder_token=placeholder_token_one,
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        use_center_crop=args.use_center_crop,
        use_random_flip=args.use_random_flip,
        set="train",
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.dataloader_num_workers
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        step_rules="1:1000,0.1:2000,0.01"
    )

    text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        vars_to_log_in_trackers = {k: str(v) if isinstance(v, (pathlib.Path)) else v for k, v in vars(args).items()}
        accelerator.init_trackers("text2image-ti-fine-tune", config=vars_to_log_in_trackers)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = args.resume_from_checkpoint.name
        else:
            # Get the most recent checkpoint
            checkpoint_dirs = get_checkpoints_directories(args.output_dir)
            path = checkpoint_dirs[-1] if len(checkpoint_dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{str(args.resume_from_checkpoint)}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(str(pathlib.Path(args.output_dir, path)))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
        
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Retrieve token embeddings from text_encoder_one to make sure  we dont update them later
    # (we only want to update the embeddings that hold the concept of <placeholder_token> and additional vectors)
    original_token_embeddings_one = get_token_embeddings(unwrap_model_(text_encoder_one))
    original_token_embeddings_two = get_token_embeddings(unwrap_model_(text_encoder_two))

    for epoch in range(first_epoch, args.num_train_epochs):
        text_encoder_one.train()
        text_encoder_two.train()
        
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(text_encoder_one):
                # Convert images to latent space
                if args.pretrained_vae_model_name_or_path is not None:
                    pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                else:
                    pixel_values = batch["pixel_values"]

                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor
                if args.pretrained_vae_model_name_or_path is None:
                    model_input = model_input.to(weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (model_input.shape[0], model_input.shape[1], 1, 1), device=model_input.device
                    )

                batch_size = model_input.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=model_input.device
                )
                timesteps = timesteps.long()

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                # Micro-Conditioning, sec. 2.2 in the paper (Fig. 3 and Fig. 4 in the paper). 
                def compute_time_ids(original_size, crops_coords_top_left):
                    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                    target_size = (args.resolution, args.resolution)
                    add_time_ids = list(original_size + crops_coords_top_left + target_size)
                    add_time_ids = torch.tensor([add_time_ids])
                    add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
                    return add_time_ids

                add_time_ids = torch.cat([
                    compute_time_ids(original_size, crop_top_size) 
                    for original_size, crop_top_size in zip(batch["original_sizes"], batch["crop_top_lefts"])
                ])

                # Fuse text embeddings from two text encoders:
                # 1. prompt_embeds are concatenated embeddings from both CLIP text encoders; 
                # 2. pooled_prompt_embeds are pooled text embedding from the OpenCLIP model; 
                # NOTE: pooled_prompt_embeds are also going into the Refiner UNet, if such model is set (here we're fine-tuning base model only)
                # (see https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0#%F0%9F%A7%A8-diffusers)
                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    text_encoders=[text_encoder_one, text_encoder_two],
                    tokenizers=None,
                    prompt=None,
                    text_input_ids_list=[batch["input_ids_one"], batch["input_ids_two"]],
                )

                prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
                pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype)

                unet_added_conditions = {"time_ids": add_time_ids, 
                                         "text_embeds": pooled_prompt_embeds}
                
                # Predict the noise residual
                model_pred = unet(
                    noisy_model_input,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs=unet_added_conditions,
                    return_dict=False,
                )[0]

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    # https://arxiv.org/abs/2202.00512
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()
                
                if args.debug_loss and "filenames" in batch:
                    for fname in batch["filenames"]:
                        accelerator.log({"loss_for_" + fname: loss}, step=global_step)
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                                
                # Attempt to zero out unwanted gradients to avoid copying, but for some reason after optimizer.step() 
                # weights still change a bit, although gradients in the optim itself are zeroes (except for placeholder tokens, ofc).         
                
                # if accelerator.sync_gradients:
                    #with torch.no_grad():   
                        # weights = unwrap_model_(text_encoder_one).text_model.embeddings.token_embedding.weight
                        # accelerator.print(f"before: {weights.grad}")
                        # weights.grad[indices_without_update, :].zero_()
                        # accelerator.print(f"after: {weights.grad}\n====")
            
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # TODO ensure we do not update token embeddings of text_encoders,
                # only the ones which hold the concept of <placeholder_token> and additional vectors
                with torch.no_grad():
                    update_placeholder_tokens_only(unwrap_model_(text_encoder_one), 
                                                   tokenizer_one, 
                                                   placeholder_token_ids_one, 
                                                   original_token_embeddings_one)
                    update_placeholder_tokens_only(unwrap_model_(text_encoder_two), 
                                                   tokenizer_two, 
                                                   placeholder_token_ids_two, 
                                                   original_token_embeddings_two)
                    
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoint_dirs = get_checkpoints_directories(args.output_dir)

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoint_dirs) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoint_dirs) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoint_dirs[:num_to_remove]

                                logger.info(
                                    f"{len(checkpoint_dirs)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = pathlib.Path(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = pathlib.Path(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompt and global_step % args.validation_steps == 0:
                        validate(
                            args, 
                            accelerator, 
                            unet, 
                            vae, 
                            unwrap_model_(text_encoder_one), 
                            unwrap_model_(text_encoder_two), 
                            tokenizer_one,
                            tokenizer_two,
                            epoch, 
                            weight_dtype)         

                if global_step % args.embeddings_save_steps == 0:
                    weight_name = f"ti-embeddings-step-{global_step}.safetensors"
                    save_path = pathlib.Path(args.output_dir, weight_name)
                    save_embeddings(
                        unwrap_model_(text_encoder_one),
                        unwrap_model_(text_encoder_two),                        
                        placeholder_token_ids_one,
                        placeholder_token_ids_two,
                        save_path,
                        safe_serialization=True,
                    )

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break        

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        validate(
            args, 
            accelerator, 
            unet, 
            vae, 
            unwrap_model_(text_encoder_one), 
            unwrap_model_(text_encoder_two), 
            tokenizer_one,
            tokenizer_two,
            epoch, 
            weight_dtype)
        
        save_path = pathlib.Path(args.output_dir, "ti-embeddings-final.safetensors")
        save_embeddings(
            unwrap_model_(text_encoder_one),
            unwrap_model_(text_encoder_two),
            placeholder_token_ids_one,
            placeholder_token_ids_two,
            save_path,
            safe_serialization=True,
        )
        
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_train_args()
    
    path_to_log_file = pathlib.Path(args.output_dir, args.logging_dir, "train.log")
    path_to_log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = get_logger(__name__, log_level="INFO", path_to_log_file=path_to_log_file)

    main(args)