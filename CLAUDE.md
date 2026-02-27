# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a PyTorch implementation of YOLOv1 (You Only Look Once) for object detection on the Pascal VOC 2007 dataset. The implementation follows the original YOLO paper with a 7×7 grid, 2 bounding boxes per grid cell, and 20 object classes.

## Setup and Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare the Pascal VOC 2007 dataset:
   - Download the VOC 2007 dataset and place it in `./data/VOCdevkit2007/VOC2007`
   - The directory structure should match:
     ```
     data/VOCdevkit2007/VOC2007/
       Annotations/
       JPEGImages/
       ImageSets/Main/
     ```
   - Alternatively, update `config.py` to point to your dataset location.

## Common Development Tasks

### Training
```bash
python train.py
```
- Uses configuration from `config.py` (via `get_train_config()`)
- Logs are saved to `./log` directory with timestamp
- Checkpoints are saved in `./checkpoints` (configurable)
- Training uses SGD with momentum, weight decay, and learning rate scheduling

### Validation
Validation runs automatically after each training epoch in `train.py`. The validation loss is computed and the best model is saved.

### Testing
No dedicated evaluation script is provided. To test the model, you can write a custom script that loads a checkpoint and runs inference.

### Running Individual Components
- **Model**: `from model import YOLOModule`
- **Dataset**: `from dataset import YOLODataset`
- **Loss**: `from loss import LossModul`
- **Transforms**: `from transforms import ToRequired, ArrayToTensor`

## Architecture Overview

### Configuration (`config.py`)
- `YOLOv1Config`: Dataclass containing all hyperparameters (grid size, bounding boxes, classes, training parameters, loss weights, data augmentation settings).
- Default values match the original YOLOv1 paper: `S=7`, `B=2`, `C=20`, `lambda_coord=5.0`, `lambda_noobj=0.5`.
- Automatically creates necessary directories on initialization.

### Model (`model.py`)
- `YOLOModule`: The YOLOv1 neural network.
  - **Feature extractor**: 24 convolutional layers with LeakyReLU activations and max pooling, matching the original architecture.
  - **Fully connected layers**: Flatten → 4096-dim → dropout → 1470-dim → reshape to `[batch, 7, 7, 30]`.
  - Output tensor format: `[batch, S, S, B*5 + C]` where each bounding box has `(x, y, w, h, confidence)` and class probabilities.
- `create_model(args)`: Factory function returning the model.

### Dataset (`dataset.py`)
- `YOLODataset`: Loads Pascal VOC annotations and images.
  - Parses XML annotation files and converts bounding boxes to YOLO format `(class_idx, x_center, y_center, width, height)` normalized to `[0,1]`.
  - Uses `ImageSets/Main/train.txt` and `val.txt` for train/validation splits.
- `get_dataloader(args)`: Returns train and validation DataLoader instances.

### Transforms (`transforms.py`)
- `ToRequired`: Converts images and YOLO annotations to YOLOv1 input format.
  - Resizes images to `448×448` with letterboxing (maintaining aspect ratio, padded with gray).
  - Scales bounding boxes accordingly.
  - Maps ground truth boxes to the `7×7` grid, producing target tensor shape `[S, S, B*5 + C]`.
- `ArrayToTensor`: Converts numpy arrays to PyTorch tensors.

### Loss (`loss.py`)
- `LossModul`: Implements the YOLOv1 loss function.
  - **Localization loss**: MSE on bounding box center coordinates, and square root of width/height (with λ_coord weighting).
  - **Confidence loss**: MSE for both object and no-object predictions (with λ_noobj weighting for no-object).
  - **Classification loss**: MSE on class probabilities.
  - Uses IoU to determine which of the two predicted boxes is responsible for each ground truth object.

## Data Structure

The project expects Pascal VOC 2007 dataset in the following structure (relative to repository root):

```
data/VOCdevkit2007/VOC2007/
  Annotations/          # XML annotation files
  JPEGImages/           # Original images
  ImageSets/Main/
    train.txt           # List of training image names (without extension)
    val.txt             # List of validation image names
```

If your dataset is located elsewhere, update the `data_dir` field in `config.py`.

## Configuration Details

Key hyperparameters in `config.py`:
- `S`, `B`, `C`: Grid size, bounding boxes per cell, number of classes.
- `lambda_coord`, `lambda_noobj`: Loss weighting factors.
- `image_size`: Input image size (default 448×448).
- `batch_size`, `epochs`, `lr`, `weight_decay`, `momentum`: Training parameters.
- `save_dir`, `log_dir`: Directories for checkpoints and TensorBoard logs.

## Additional Files

### Test Directory (`test/`)
The `test/` directory contains experimental or alternative implementations:
- `test/config.py`: Slightly different configuration with additional training options.
- `test/train.py`: More comprehensive training script with command-line arguments, checkpointing, learning rate scheduling, and TensorBoard logging.
- `test/utils.py`: Utility functions for visualization, evaluation (mAP calculation), model export, and loss plotting.
- `test/test.py`: Minimal test file.

These files are not integrated into the main training pipeline but can be used as reference or extended.

## Notes

- The implementation is designed for Pascal VOC 2007 with 20 classes. To use a different dataset, update `class_names` in `config.py` and ensure annotations follow Pascal VOC XML format.
- Training uses CPU by default (`device='cpu'`). To use GPU, change `device` in `config.py` to `'cuda'`.
- The loss function expects target tensors where both bounding boxes in the same grid cell have identical values (or zeros if no object).
- The `train.py` script has hard-coded epochs and device settings; these override config values. Consider modifying the script to use config fully.
- For visualization and evaluation, refer to `test/utils.py` for functions like `visualize_predictions`, `plot_loss_curve`, and `calculate_mAP` (skeleton).