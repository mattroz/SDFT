# Adapted from HuggingFace diffusers scripts
# https://github.com/huggingface/diffusers

import cv2
import pathlib

import datasets
import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed

import diffusers
from diffusers import (
    AutoencoderKL,
    StableDiffusionXLPipeline,
)

from src.utils.logger import get_logger
from src.methods.lora.arguments import parse_inference_args


logger = get_logger(__name__, log_level="INFO", path_to_log_file="logs/inference.log") 


def main(args):
    logging_dir = pathlib.Path(args.output_dir, args.logging_dir)
    path_to_save_images = pathlib.Path(args.output_dir, "inference_results")
    path_to_save_images.mkdir(parents=True, exist_ok=True)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

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

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if args.mixed_precision == "fp16":
        vae.to(weight_dtype)
    
    logger.info(f"Loading pipeline in {weight_dtype} precision", main_process_only=True)

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    ).to(accelerator.device)

    # load LoRA weights and inject them to cross-attention in UNet
    pipeline.load_lora_weights(args.lora_checkpoint_path, adapter_name="dark_fantasy")

    # actually do not need this here, but in case you want to merge multiple LoRAs - here you go
    pipeline.set_adapters(["dark_fantasy"], adapter_weights=[1.0])
    
    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    for i in range(args.num_validation_images):
        # TODO add negative prompt
        image = pipeline(args.validation_prompt, 
                         num_inference_steps=35,
                         guidance_scale=5.0,
                         generator=generator).images[0]
        path_to_save_image = pathlib.Path(path_to_save_images, 
                                          "_".join(args.validation_prompt.lower().replace(",","").replace(".", "").split(" "))+f"_{i}.png")
        image = np.asarray(image)
        cv2.imwrite(str(path_to_save_image), image)
        

if __name__ == "__main__":
    args = parse_inference_args()
    main(args)
