# Skull Textual Inversion Dataset

This toy dataset contains 5 images for fine-tuning text-to-image model with textual inversion approach.

This version of a dataset does not contain any captions as they're taken from a [dataset]() implementation according to the TI paper. 

## Dataset Details

- Number of images: 5
- Image resolution: 1024x1024

## Data Collection

The images were arbitrarily taken on phone camera and cropped afterwards. 

Since default captions themselves may not quite accurately describe the style or object presented, TI embeddings tend to overfit. Solution: give more meainigful and precise captions.

## Dataset Format

The dataset is provided in a following format:
- `images/` directory
