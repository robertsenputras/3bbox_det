# 3D Object Detection Pipeline with YOLOv11-seg and Frustum-PointNet

This repository implements a 3D object detection pipeline that combines YOLOv11-seg for 2D detection and segmentation with Frustum-PointNet for 3D box estimation.

## Pipeline Overview

The pipeline consists of two main stages:

1. 2D Stage:
   - Uses YOLOv11-seg for class-agnostic object detection and segmentation
   - Provides 2D bounding boxes and segmentation masks

2. 3D Stage:
   - Lifts each 2D detection to a frustum using depth information
   - Uses Frustum-PointNet to estimate 3D bounding boxes
   - Combines results and applies 3D non-maximum suppression

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/3d-detection-pipeline.git
cd 3d-detection-pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pre-trained weights:
   - YOLOv11-seg weights should be placed in the root directory
   - Frustum-PointNet weights should be downloaded from the original repository

## Usage

Run the detection pipeline on a single image:

```bash
python detect.py \
    --image path/to/image.jpg \
    --depth path/to/depth.npy \
    --calib path/to/calib.txt \
    --yolo-weights path/to/yolo11m-seg.pt \
    --frustum-weights path/to/frustum_pointnet.pth \
    --output path/to/output.jpg
```

### Input Format

- Image: RGB image in standard formats (jpg, png)
- Depth: Numpy array (.npy) containing depth values
- Calibration: 3x4 camera projection matrix in text format

### Output Format

The pipeline outputs:
- 3D bounding boxes in camera coordinates
- Confidence scores for each detection
- Visualization of results (if output path is provided)

## Visualization

The visualization includes:
1. 2D bounding boxes on the RGB image
2. Projected 3D boxes on the RGB image
3. Bird's eye view visualization of the scene

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yolov11seg,
  title={YOLOv11-seg: Real-time Instance Segmentation},
  author={...},
  journal={...},
  year={2023}
}

@inproceedings{qi2018frustum,
  title={Frustum pointnets for 3d object detection from rgb-d data},
  author={Qi, Charles R and Liu, Wei and Wu, Chenxia and Su, Hao and Guibas, Leonidas J},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={918--927},
  year={2018}
}
``` 