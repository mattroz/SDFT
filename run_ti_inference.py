import cv2
import datetime
import torch
import pathlib
import transformers
import diffusers

import numpy as np

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed

from diffusers import AutoencoderKL, StableDiffusionXLPipeline
from diffusers.utils.torch_utils import randn_tensor
from safetensors.torch import load_file

from src.utils.logger import get_logger
from src.methods.textual_inversion.arguments import parse_inference_args


def generate_and_save(args, accelerator, pipeline, latents, path_to_save_images):
    image = pipeline(
        latents=latents,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,             
        prompt=args.prompt,
        prompt_2=args.prompt_2,
        negative_prompt=args.negative_prompt,
        negative_prompt_2=args.negative_prompt_2
    ).images[0]
    
    if accelerator.is_main_process:
        ts_now = datetime.datetime.now().isoformat(timespec="seconds")
        path_to_save_image = pathlib.Path(path_to_save_images, 
                                        "_".join(args.prompt.lower().replace(",","").replace(".", "").split(" "))+f"_{ts_now}.png")
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(path_to_save_image), image)


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
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
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
    
    embeddings_state_dict = load_file(args.path_to_embeddings)
    
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    latents_shape = (1, 
                     pipeline.unet.config.in_channels, 
                     args.resolution // pipeline.vae_scale_factor, 
                     args.resolution // pipeline.vae_scale_factor)
    latents = randn_tensor(latents_shape, generator, device=accelerator.device, dtype=weight_dtype)

    # Load token embeddings for text encoder one and text encoder two
    pipeline.load_textual_inversion(
        embeddings_state_dict["text_encoder_one"], 
        token=args.placeholder_token,
        text_encoder=pipeline.text_encoder, 
        tokenizer=pipeline.tokenizer)
    
    pipeline.load_textual_inversion(
        embeddings_state_dict["text_encoder_two"], 
        token=args.placeholder_token, 
        text_encoder=pipeline.text_encoder_2, 
        tokenizer=pipeline.tokenizer_2)
    
    for _ in range(args.num_images_to_generate):
        # latents = randn_tensor(latents_shape, generator, device=accelerator.device, dtype=weight_dtype)
        generate_and_save(args, accelerator, pipeline, latents, path_to_save_images)


if __name__ == "__main__":
    args = parse_inference_args()
    
    path_to_log_file = pathlib.Path(args.output_dir, args.logging_dir, "ti_inference.log")
    logger = get_logger(__name__, log_level="INFO", path_to_log_file=path_to_log_file) 

    main(args)