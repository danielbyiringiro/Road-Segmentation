# Road Segmentation

A deep learning project for semantic segmentation of roads in images using a VGG-FCN8 architecture with PyTorch.

## Overview

This project implements road segmentation using a Fully Convolutional Network (FCN) based on VGG16 encoder. The model is trained on the KITTI Road Segmentation dataset to perform binary semantic segmentation, distinguishing between road and non-road pixels in images.

## Model Architecture

The project uses **VGG-FCN8**, a fully convolutional network that consists of:

- **Encoder**: Pre-trained VGG16 feature extractor (blocks 3, 4, and 5)
- **Decoder**: Upsampling layers with skip connections for fine-grained segmentation
- **Output**: Binary segmentation mask (road vs. non-road)

## Dataset

The model is trained on the [KITTI Road Segmentation Dataset](https://www.kaggle.com/datasets/sakshaymahna/kittiroadsegmentation), which contains:

- **Training images**: 289 RGB images of road scenes
- **Ground truth**: Corresponding binary segmentation masks
- **Image size**: Resized to 128x128 for training

The dataset is split as follows:
- Training: 80% (231 images)
- Validation: 10% (28 images)  
- Testing: 10% (30 images)

## Requirements

- Python 3.8+
- PyTorch >= 1.10.0
- torchvision >= 0.11.0
- torchmetrics
- matplotlib
- numpy
- jupyter (for running the notebook)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/danielbyiringiro/Road-Segmentation.git
cd Road-Segmentation
```

2. Install the required dependencies:
```bash
pip install torch torchvision torchmetrics matplotlib numpy jupyter
```

Or create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install torch torchvision torchmetrics matplotlib numpy jupyter
```

3. Download the KITTI Road Segmentation dataset:

**Note**: You'll need Kaggle API credentials to download the dataset. Set up your Kaggle API token first:
```bash
# Place your kaggle.json in ~/.kaggle/
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Then download the dataset:
```bash
pip install kaggle
kaggle datasets download -d sakshaymahna/kittiroadsegmentation
unzip kittiroadsegmentation.zip
```

Alternatively, download manually from [Kaggle](https://www.kaggle.com/datasets/sakshaymahna/kittiroadsegmentation) and extract to the project directory.

## Usage

The project is implemented as a Jupyter notebook. To run it:

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `Road Segmentation.ipynb`

3. Run the cells sequentially to:
   - Import libraries
   - Load and prepare the dataset
   - Define the VGG-FCN8 model
   - Train the model
   - Evaluate performance

## Training Configuration

- **Image Size**: 128x128
- **Batch Size**: 32
- **Loss Function**: Binary Cross-Entropy (BCE)
- **Optimizer**: Adam with learning rate of 1e-4
- **Evaluation Metric**: Binary Jaccard Index (IoU)

## Project Structure

```
Road-Segmentation/
├── Road Segmentation.ipynb    # Main notebook with complete implementation
├── README.md                   # Project documentation
├── training/
│   ├── image_2/               # Training images
│   └── gt_image_2/            # Ground truth masks
└── testing/                   # Test images
```

## Features

- Custom PyTorch Dataset implementation for road segmentation
- VGG16-based FCN architecture with skip connections
- Binary semantic segmentation
- IoU metric for performance evaluation
- Data augmentation and preprocessing

## License

This project is open source and available for educational and research purposes.

## Acknowledgments

- KITTI Vision Benchmark Suite for providing the dataset
- VGG and FCN architectures from the respective research papers
