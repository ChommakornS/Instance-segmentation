# Instance Segmentation with Detectron2

This repository contains code for training and evaluating instance segmentation models using Facebook's Detectron2 framework. The implementation demonstrates how to prepare custom datasets from Open Images, train Mask R-CNN models, and perform instance segmentation on images.

## Project Overview

This project implements instance segmentation, which involves:
1. Detecting objects in an image
2. Classifying each detected object
3. Generating pixel-precise masks for each instance

The implementation uses Detectron2, a powerful object detection and segmentation library by Facebook AI Research. The project focuses on segmenting three specific classes: Panda, Dice, and Duck.

## Dataset Preparation

The dataset for this project was prepared using FiftyOne, which facilitated:
- Loading images from Open Images V7 dataset
- Filtering for specific classes (Panda, Dice, Duck)
- Selecting and tagging a subset for annotation
- Exporting data in COCO format for compatibility with Detectron2

### Dataset Statistics
- Source: Open Images V7 validation split
- Classes: Panda, Dice, Duck
- Format: COCO Detection Dataset with instance segmentation masks
- Number of samples: 30 selected images from the original dataset

## Setup Instructions

### Prerequisites
- Python 3.7+
- PyTorch 1.7+
- CUDA-capable GPU (recommended)
- Google Colab (for running the notebooks)

### Installation

The project is implemented as Jupyter notebooks designed to run in Google Colab:

1. Access the full implementation here:
   [Instance Segmentation with Detectron2 Code](https://drive.google.com/file/d/1-OuPgBj0HxCuqzU_GdvyNPg7TCoilRLM/view?usp=drive_link)

2. Install required dependencies in your Colab environment:
```python
!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html
!pip install fiftyone
```

### Dataset Preparation

1. Use the provided `Prepare Dataset with FiftyOne.ipynb` notebook to:
   - Download and filter images from Open Images V7
   - Select and tag a subset for annotation using FiftyOne's interactive UI
   - Export in COCO format compatible with Detectron2

2. The dataset is structured in COCO format:
```
data_l2/
├── images/
└── labels.json
```

## Model Training

The implementation uses a Mask R-CNN model with ResNet-50-FPN backbone pretrained on COCO dataset:

```python
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
```

Training configuration:
- Learning rate: 0.00025
- Batch size: 2
- Maximum iterations: 300
- Images per batch: 2
- Classes: ['Panda', 'Dice', 'Duck']

## Inference

Inference is performed using the trained model with visualization of results:

```python
predictor = DefaultPredictor(cfg)
outputs = predictor(img)
v = Visualizer(img[:, :, ::-1], metadata, scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
```

## Results

The model successfully detects and segments instances of Panda, Dice, and Duck with high precision. The implementation demonstrates effective transfer learning by fine-tuning a pre-trained Mask R-CNN model on a small custom dataset.

You can view the full output and performance metrics in the project outputs:
[View Model Output and Results](https://drive.google.com/file/d/1-kUcc0gsxXmvZgcN78BFEmOXC8uyABwG/view?usp=sharing)

## Visualization

The visualization includes:
- Colored instance masks overlaid on the original images
- Bounding boxes around detected objects
- Class labels and confidence scores
- Each instance represented with a unique color

Example output visualizations demonstrate the model's ability to precisely segment instances even in challenging scenes with multiple objects and overlapping instances.

## Implementation Details

The implementation follows these key steps:

1. **Dataset Preparation**:
   - Downloading images with segmentation masks from Open Images
   - Filtering for specific classes (Panda, Dice, Duck)
   - Converting to COCO format for Detectron2 compatibility

2. **Model Configuration**:
   - Using Mask R-CNN with ResNet-50 backbone
   - Configuring for 3 classes
   - Setting training hyperparameters

3. **Training**:
   - Fine-tuning the pre-trained model
   - Monitoring training loss
   - Evaluating performance on validation set

4. **Evaluation & Visualization**:
   - Running inference on test images
   - Visualizing instance masks with colored overlays
   - Displaying class labels and confidence scores

## Key Features

- Transfer learning using pre-trained models
- Instance-level segmentation (not just semantic segmentation)
- Integration with FiftyOne for dataset curation
- COCO format compatibility

## Acknowledgments

- Facebook AI Research for creating Detectron2
- Open Images V7 for the dataset
- FiftyOne for dataset preparation utilities

## References

- [Detectron2 GitHub](https://github.com/facebookresearch/detectron2)
- [Mask R-CNN Paper](https://arxiv.org/abs/1703.06870)
- [FiftyOne Documentation](https://voxel51.com/docs/fiftyone/)
- [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html)
