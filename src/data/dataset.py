# Adapted from HuggingFace diffusers scripts
# https://github.com/huggingface/diffusers

# TODO pass intermediate values between functions in a config rather than as cmd_arguments

import os
import torch
import random

import numpy as np

from torchvision import transforms
from torchvision.transforms.functional import crop
from datasets import load_dataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    original_sizes = [example["original_sizes"] for example in examples]
    crop_top_lefts = [example["crop_top_lefts"] for example in examples]
    input_ids_one = torch.stack([example["input_ids_one"] for example in examples])
    input_ids_two = torch.stack([example["input_ids_two"] for example in examples])
    result = {
        "pixel_values": pixel_values,
        "input_ids_one": input_ids_one,
        "input_ids_two": input_ids_two,
        "original_sizes": original_sizes,
        "crop_top_lefts": crop_top_lefts,
    }

    filenames = [example["filenames"] for example in examples if "filenames" in example]
    if filenames:
        result["filenames"] = filenames
    return result


def build_dataset(args):
    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir, data_dir=args.train_data_dir
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
    
    return dataset


def tokenize_captions(examples, tokenizers, caption_column, is_train=True):
    captions = []
    for caption in examples[caption_column]:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # If multiple captions provided, get random one.
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"Caption column `{caption_column}` should contain either strings or lists of strings."
            )
        
    # We have two tokenizers here for SDXL.
    tokenized_captions = []
    for tokenizer in tokenizers:
        tokenized_caption = tokenizer(captions, 
                                      padding="max_length", 
                                      max_length=tokenizer.model_max_length, 
                                      truncation=True, 
                                      return_tensors="pt").input_ids
        tokenized_captions.append(tokenized_caption)
    
    assert len(tokenized_captions) == len(tokenizers) == 2
    
    return tokenized_captions


def build_train_processing_fn(args, dataset, tokenizers, dataset_columns=None):
    column_names = dataset["train"].column_names

    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
        
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )
    
    # Set of augmentations (TODO: Albumentations integration?)
    train_resize = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    train_crop = transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution)
    train_flip = transforms.RandomHorizontalFlip(p=1.0)
    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    
    # Use builder pattern, because there is no way to provide all the data needed throught the arguments on runtime,
    def train_processing_fn(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        # image aug
        original_sizes = []
        all_images = []
        crop_top_lefts = []
        for image in images:
            original_sizes.append((image.height, image.width))
            image = train_resize(image)
            if args.random_flip and random.random() < 0.5:
                # flip
                image = train_flip(image)
            if args.center_crop:
                y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
                x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(image, (args.resolution, args.resolution))
                image = crop(image, y1, x1, h, w)
            crop_top_left = (y1, x1)
            crop_top_lefts.append(crop_top_left)
            image = train_transforms(image)
            all_images.append(image)

        examples["original_sizes"] = original_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        examples["pixel_values"] = all_images
        examples["input_ids_one"], examples["input_ids_two"] = \
            tokenize_captions(examples, tokenizers, caption_column)
        
        if args.debug_loss:
            fnames = [os.path.basename(image.filename) for image in examples[image_column] if image.filename]
            if fnames:
                examples["filenames"] = fnames
        return examples
    
    return train_processing_fn