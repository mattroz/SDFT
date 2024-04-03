# Adapted from HuggingFace diffusers scripts
# https://github.com/huggingface/diffusers

# TODO pass intermediate values between functions in a config rather than as cmd_arguments

import os
import torch
import random
import PIL
import cv2
import pathlib

import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from datasets import load_dataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    original_sizes = [example["original_size"] for example in examples]
    crop_top_lefts = [example["crop_top_left"] for example in examples]
    input_ids_one = torch.stack([example["input_id_one"] for example in examples])
    input_ids_two = torch.stack([example["input_id_two"] for example in examples])
    result = {
        "pixel_values": pixel_values,
        "input_ids_one": input_ids_one,
        "input_ids_two": input_ids_two,
        "original_sizes": original_sizes,
        "crop_top_lefts": crop_top_lefts,
    }

    filenames = [example["filename"] for example in examples if "filename" in example]
    if filenames:
        result["filenames"] = filenames
    return result


def tokenize_caption(caption, tokenizers, is_train=True):
    if isinstance(caption, str):
        caption = [caption]
    elif isinstance(caption, (list, np.ndarray)):
        # If multiple captions provided, get random one.
        caption = [random.choice(caption) if is_train else caption[0]]
    else:
        raise ValueError(
            f"Caption `{caption}` should be either string or lists of strings."
        )
        
    # We have two tokenizers here for SDXL.
    tokenized_captions = []
    for tokenizer in tokenizers:
        tokenized_caption = tokenizer(caption, 
                                      padding="max_length", 
                                      max_length=tokenizer.model_max_length, 
                                      truncation=True, 
                                      return_tensors="pt").input_ids
        tokenized_captions.append(tokenized_caption)
    
    assert len(tokenized_captions) == len(tokenizers) == 2
    
    return tokenized_captions


# LORA DATASET
class ImageCaptionDataset(Dataset):
    def __init__(self,
                 tokenizer_one,
                 tokenizer_two,
                 image_column=None,
                 caption_column="text",
                 train_data_dir=None,
                 dataset_name=None,
                 dataset_config_name=None,
                 cache_dir=None,
                 resolution=512,
                 use_center_crop=True,
                 use_random_flip=True,
                 max_train_samples=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 shuffle=True,
                 seed=42,
                 debug=False):
        self.resolution = resolution
        self.use_random_flip = use_random_flip
        self.use_center_crop = use_center_crop
        self.interpolation = interpolation
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.debug = debug

        if dataset_name is not None:
            self.dataset = load_dataset(
                dataset_name, 
                dataset_config_name, 
                cache_dir=cache_dir, 
                data_dir=train_data_dir
            )
        else:
            data_files = {}
            if train_data_dir is not None:
                data_files["train"] = [str(path) for path in pathlib.Path(train_data_dir).glob("**/*") if path.is_file()]
            self.dataset = load_dataset("imagefolder", data_files=data_files, cache_dir=cache_dir)
    
        column_names = self.dataset["train"].column_names

        if image_column is None:
            self.image_column = column_names[0]
        else:
            self.image_column = image_column
            if image_column not in column_names:
                raise ValueError(
                    f"--image_column' value '{image_column}' needs to be one of: {', '.join(column_names)}"
                )
            
        if caption_column is None:
            self.caption_column = column_names[1]
        else:
            self.caption_column = caption_column
            if caption_column not in column_names:
                raise ValueError(
                    f"--caption_column' value '{caption_column}' needs to be one of: {', '.join(column_names)}"
                )
        
        if shuffle:
            self.dataset["train"] = self.dataset["train"].shuffle(seed=seed)

        if max_train_samples:
            self.dataset["train"] = self.dataset["train"].select(range(max_train_samples))

        self.train_resize = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)
        self.train_crop = transforms.CenterCrop(resolution) if use_center_crop else transforms.RandomCrop(resolution)
        self.train_flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.dataset["train"])

    def __getitem__(self, index):
        example = self.dataset["train"][index]
        
        image = example[self.image_column].convert("RGB")
        # image aug
        
        original_size = (image.height, image.width)
        image = self.train_resize(image)
        if self.use_random_flip and random.random() < 0.5:
            image = self.train_flip(image)
        if self.use_center_crop:
            y1 = max(0, int(round((image.height - self.resolution) / 2.0)))
            x1 = max(0, int(round((image.width - self.resolution) / 2.0)))
            image = self.train_crop(image)
        else:
            y1, x1, h, w = self.train_crop.get_params(image, 
                                                      (self.resolution, self.resolution))
            image = crop(image, y1, x1, h, w)

        crop_top_left = (y1, x1)
        image = self.train_transforms(image)

        example["original_size"] = original_size
        example["crop_top_left"] = crop_top_left
        example["pixel_values"] = image
        example["input_id_one"], example["input_id_two"] = \
            tokenize_caption(example[self.caption_column], [self.tokenizer_one, self.tokenizer_two])
        
        if self.debug and example[self.image_column].filename:
            example["filename"] = os.path.basename(example[self.image_column].filename)

        return example