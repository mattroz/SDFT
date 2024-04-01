# Datasets for SDXL Fine-Tuning

To fine-tune the SDXL model on your own dataset, please follow the instructions below:

1. Create a new directory here (under the `datasets/` directory).
2. Name the new directory according to your dataset's name.

Inside the newly created dataset directory, ensure the following structure:

- `images/`: This directory should contain image data for finetuning.
- `metadata.jsonl` or `metadata.csv`: File, containing text captions for each image in `images/` directory.

Note, that `metadata` file is required to have two columns: `file_name` and `text`.

For a more detailed instruction refer to [HuggingFace official documentation](https://huggingface.co/docs/datasets/image_dataset#create-an-image-dataset) on image dataset creation.
