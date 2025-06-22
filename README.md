# 3D Object Detection with Frustum-PointNets

This repository contains an implementation of 3D object detection using Frustum-PointNets. The system takes RGB images and depth data as input and outputs 3D bounding boxes for detected objects.

![Demo of 3D Object Detection](demo.png)

## Overview

This project was developed as a solution to the 3D Bounding Box Detection challenge. The goal is to create an end-to-end deep learning pipeline for predicting 3D bounding boxes from RGB-D data.

### Problem Statement
Given RGB images, point clouds, and instance segmentation masks, predict accurate 3D bounding boxes for objects in the scene.

### Solution Approach
The project implements a two-stage 3D object detection pipeline:
1. 2D object detection using pre-trained YOLOv11 to generate frustums
2. 3D object detection using Frustum-PointNet to predict 3D bounding boxes

### Architecture Overview
![Architecture Diagram](architecture.png)

Key components:
- **2D Detection**: YOLOv11 for initial object detection
- **Frustum Generation**: Projects 2D detections into 3D space
- **Point Cloud Processing**: PointNet-based architecture for 3D understanding
- **3D Box Prediction**: Multi-task learning for box parameters

### Model Details
- Base Architecture: Frustum-PointNet
- Parameters: ~45M (well within 100M limit)
- Framework: PyTorch
- Key Libraries: 
  - ultralytics (YOLOv11)
  - torch
  - numpy
  - opencv-python

### Performance Metrics
The model is evaluated using:
- 3D IoU (Intersection over Union)
- Bird's eye view IoU
- Average Precision (AP) at different IoU thresholds
- Segmentation accuracy

## Installation

### Prerequisites
- Python 3.7+
- CUDA-capable GPU
- Linux environment

### Setup
1. Clone the repository:
```bash
git clone https://github.com/robertsenputras/3bbox_det.git
cd 3bbox_det
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

### Download Dataset
The raw dataset should be placed in `data/dl_challenge` with the following structure:
```
data/dl_challenge/
├── <scene_id_1>/
│   ├── rgb.jpg
│   ├── depth.npy
│   ├── mask.npy
│   └── bbox3d.npy
├── <scene_id_2>/
...
```

### Optional: Prepare YOLO Dataset
If you want to create a YOLO format dataset (not required for inference as we use pre-trained YOLO):
```bash
python create_yolo_dataset.py
```

## Training Frustum-PointNet

### Configuration
1. Navigate to the frustum_detection directory:
```bash
cd frustum_detection
```

2. Modify the training configuration in `configs/train_config.yaml`:
```yaml
# Model configuration
model_version: 'v1'
pretrained_weights: null

# Data configuration
data_root: '../data/dl_challenge'
train_val_split: 0.8
num_points: 2048

# Training configuration
batch_size: 8
num_epochs: 50
learning_rate: 0.0001
weight_decay: 0.0001
```

### Start Training
1. Train the model from the frustum_detection directory:
```bash
python train.py \
    --cfg configs/train_config.yaml \
    --output_dir ../runs/training
```

2. Monitor training progress:
- Training logs are saved in `runs/training/training.log`
- TensorBoard visualizations are available in `runs/training`

### Training Logs
Example training metrics:
```
Epoch [30/50]
Loss: 0.245
Seg Accuracy: 0.891
3D IoU: 0.756
```

## Testing

### Evaluate Frustum-PointNet
From the frustum_detection directory, run evaluation:
```bash
python test_3d.py --config configs/test_config.yaml --visualize --weights ../ckpt/frustrum_2206_1007.pth
```

### Run Complete Pipeline
To run the complete detection pipeline (YOLO + Frustum-PointNet) on new data:
```bash
python unified_detection.py --config.yaml
```

## Model Architecture Details

The Frustum-PointNet architecture consists of three main components:
1. **3D Instance Segmentation PointNet**
   - Segments points in frustum to foreground/background
   - Uses PointNet backbone for point feature extraction
   - Multi-layer perceptron for point-wise classification
   
2. **T-Net**
   - Estimates center of object for translation normalization
   - Global feature aggregation through max pooling
   - Regression head for center prediction
   
3. **Box Estimation PointNet**
   - Predicts 3D bounding box parameters (center, size, heading)
   - Leverages both local and global point features
   - Multi-task learning for different box parameters

### Loss Functions
- Segmentation: Binary Cross Entropy
- Center Estimation: Smooth L1 Loss
- Box Parameters: Combination of classification and regression losses
- Overall Loss: Weighted sum of individual losses

### Optimization
- Optimizer: Adam
- Learning Rate Schedule: Step decay
- Data Augmentation: Random flip, rotation, and point dropout

## Future Improvements

Potential enhancements that could be implemented:
1. Model optimization:
   - ONNX conversion
   - TensorRT integration
   - Model quantization
2. Architecture improvements:
   - Transformer-based point feature extraction
   - Multi-scale feature fusion
   - Attention mechanisms
3. Training optimizations:
   - Mixed precision training
   - Distributed training support
   - Curriculum learning

## Citation

If you use this code in your research, please cite:

```
@inproceedings{qi2018frustum,
  title={Frustum PointNets for 3D Object Detection from RGB-D Data},
  author={Qi, Charles R and Liu, Wei and Wu, Chenxia and Su, Hao and Guibas, Leonidas J},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2018}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.