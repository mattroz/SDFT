# Adapted from HuggingFace diffusers scripts
# https://github.com/huggingface/diffusers

import torch

from functools import partial

from peft import set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict

from diffusers import StableDiffusionXLPipeline
from diffusers.loaders import LoraLoaderMixin
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder, 
    cast_training_params
)

from diffusers.utils import (
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
)

from src.utils.hf_helpers import unwrap_model


# TODO throw project config instead of multiple arguments

# Using builder patter here, because hooks refer to `accelerator` and other arguments,
# which cannot be provided on runtime.
def build_save_model_hook(accelerator, unet, text_encoder_one, text_encoder_two):
    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format.
    
    unwrap_model_ = partial(unwrap_model, accelerator)

    def save_model_hook(models, weights, output_dir):
        # if accelerator.is_main_process:
            
        # there are only two options here. Either are just the unet attn processor layers
        # or there are the unet and text encoder attn layers
        unet_lora_layers_to_save = None
        text_encoder_one_lora_layers_to_save = None
        text_encoder_two_lora_layers_to_save = None

        for model in models:
            if isinstance(unwrap_model_(model), type(unwrap_model_(unet))):
                unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
            elif isinstance(unwrap_model_(model), type(unwrap_model_(text_encoder_one))):
                text_encoder_one_lora_layers_to_save = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(model)
                )
            elif isinstance(unwrap_model_(model), type(unwrap_model_(text_encoder_two))):
                text_encoder_two_lora_layers_to_save = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(model)
                )
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

            # make sure to pop weight so that corresponding model is not saved again
            if weights:
                weights.pop()

        StableDiffusionXLPipeline.save_lora_weights(
            output_dir,
            unet_lora_layers=unet_lora_layers_to_save,
            text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
            text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
        )
    
    return save_model_hook


def build_load_model_hook(accelerator, args, logger, unet, text_encoder_one, text_encoder_two):

    unwrap_model_ = partial(unwrap_model, accelerator)

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model_(unet))):
                unet_ = model
            elif isinstance(model, type(unwrap_model_(text_encoder_one))):
                text_encoder_one_ = model
            elif isinstance(model, type(unwrap_model_(text_encoder_two))):
                text_encoder_two_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, _ = LoraLoaderMixin.lora_state_dict(input_dir)
        unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        if args.train_text_encoder:
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_)

            _set_state_dict_into_text_encoder(
                lora_state_dict, prefix="text_encoder_2.", text_encoder=text_encoder_two_
            )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [unet_]
            if args.train_text_encoder:
                models.extend([text_encoder_one_, text_encoder_two_])
            cast_training_params(models, dtype=torch.float32)
    
    return load_model_hook