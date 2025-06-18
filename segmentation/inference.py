# inference.py using ultralytics to 19June_0022.pt

import os
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.ops import nms
from ultralytics.engine.results import Results, Boxes

def preprocess_image(image, target_size=640):
    """
    Preprocess image to 640x640 while maintaining aspect ratio with padding.
    
    Args:
        image: Input image
        target_size: Target size for both height and width
    
    Returns:
        preprocessed_image: Resized and padded image
        scale: Scale factor used for resizing
        pad: (top_pad, left_pad) padding values
    """
    # Get original dimensions
    height, width = image.shape[:2]
    
    # Calculate scale to maintain aspect ratio
    scale = min(target_size / width, target_size / height)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Create square canvas with padding
    square = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # Calculate padding
    top_pad = (target_size - new_height) // 2
    left_pad = (target_size - new_width) // 2
    
    # Place the resized image on the square canvas
    square[top_pad:top_pad + new_height, left_pad:left_pad + new_width] = resized
    
    return square, scale, (top_pad, left_pad)

def visualize_prediction(image, results, save_path=None, conf_threshold=0.25, original_size=None, scale=None, pad=None):
    """
    Visualize the prediction results on the image.
    
    Args:
        image: Original image
        results: YOLO prediction results
        save_path: Path to save the visualization
        conf_threshold: Confidence threshold for displaying predictions
        original_size: Original image size (height, width)
        scale: Scale factor used in preprocessing
        pad: (top_pad, left_pad) used in preprocessing
    """
    # Create a copy of the image for visualization
    vis_image = image.copy()
    
    # Get the predictions
    masks = results[0].masks
    boxes = results[0].boxes
    
    if masks is None:
        print("No detections found!")
        return
    
    # Generate random colors for each instance
    colors = np.random.randint(0, 255, size=(len(masks), 3), dtype=np.uint8)
    
    # Draw each instance
    for i, (mask, box) in enumerate(zip(masks, boxes)):
        confidence = float(box.conf)
        if confidence < conf_threshold:
            continue
            
        # Get the binary mask
        binary_mask = mask.data.cpu().numpy()[0]
        
        # If we need to restore original size
        if original_size is not None and scale is not None and pad is not None:
            # Remove padding
            top_pad, left_pad = pad
            binary_mask = binary_mask[top_pad:top_pad + int(original_size[0] * scale),
                                    left_pad:left_pad + int(original_size[1] * scale)]
            # Resize back to original size
            binary_mask = cv2.resize(binary_mask.astype(np.uint8), 
                                   (original_size[1], original_size[0]), 
                                   interpolation=cv2.INTER_LINEAR) > 0.5
        
        # Create a colored overlay for the mask
        color_mask = np.zeros_like(vis_image, dtype=np.uint8)
        color_mask[binary_mask > 0] = colors[i]
        
        # Blend the mask with the image
        alpha = 0.5
        vis_image = cv2.addWeighted(vis_image, 1, color_mask, alpha, 0)
        
        # Draw bounding box (adjust coordinates if needed)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if original_size is not None and scale is not None and pad is not None:
            # Remove padding and rescale coordinates
            x1 = int((x1 - left_pad) / scale)
            y1 = int((y1 - top_pad) / scale)
            x2 = int((x2 - left_pad) / scale)
            y2 = int((y2 - top_pad) / scale)
            
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), colors[i].tolist(), 2)
        
        # Add confidence score
        label = f"Object {i+1}: {confidence:.2f}"
        cv2.putText(vis_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i].tolist(), 2)
    
    # Display the result
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def filter_detections(boxes, scores, cls, masks, 
                     conf_threshold=0.25,
                     max_box_ratio=0.8,  # Max box size relative to image
                     min_mask_ratio=0.1):  # Min mask area relative to box area
    """
    Filter detections based on multiple criteria.
    
    Args:
        boxes: Tensor of bounding boxes (N, 4) in xyxy format
        scores: Tensor of confidence scores (N)
        cls: Tensor of class IDs (N)
        masks: Tensor of segmentation masks
        conf_threshold: Minimum confidence score
        max_box_ratio: Maximum allowed box area ratio compared to image
        min_mask_ratio: Minimum mask area ratio compared to box area
    
    Returns:
        keep_indices: Indices of detections to keep
    """
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long)
    
    keep_indices = []
    
    # Get image dimensions from the mask shape
    img_height, img_width = masks.data.shape[-2:]
    
    for i, (box, score, mask) in enumerate(zip(boxes, scores, masks.data)):
        # Check confidence threshold
        if score < conf_threshold:
            print(f"Skipping detection {i} with confidence {score}")
            continue
        
        # Convert box coordinates to CPU for checking
        x1, y1, x2, y2 = box.cpu()
        
        # Check if any corner of the bounding box is outside image boundaries
        if x1 <= 0 or y1 <= 0 or x2 >= img_width or y2 >= img_height:
            print(f"Skipping detection {i} with box outside image bounds: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}) vs image ({img_width}, {img_height})")
            continue
            
        # Calculate box area and dimensions in normalized coordinates
        box_width = (x2 - x1) / img_width
        box_height = (y2 - y1) / img_height
        box_area = box_width * box_height  # This is now a ratio of image area
        
        # Check if box is too large relative to image
        if box_area > max_box_ratio:
            print(f"Skipping detection {i} with normalized box area {box_area:.3f} (max allowed: {max_box_ratio})")
            continue
        
        # Convert mask to numpy and get the correct shape
        mask_np = mask.cpu().numpy()
        
        # Ensure mask is in the correct shape (H, W)
        if len(mask_np.shape) == 3:
            binary_mask = mask_np[0] > 0  # Take first channel if multi-channel
        else:
            binary_mask = mask_np > 0
        
        # Convert box coordinates to integers and ensure they're within image bounds
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        x1, x2 = max(0, x1), min(img_width, x2)
        y1, y2 = max(0, y1), min(img_height, y2)
        
        # Calculate areas
        box_area_pixels = (x2 - x1) * (y2 - y1)
        
        # Count mask pixels within the box region
        mask_area_pixels = np.sum(binary_mask[y1:y2, x1:x2])
        
        # Calculate mask to box ratio
        mask_box_ratio = mask_area_pixels / box_area_pixels if box_area_pixels > 0 else 0
        
        # Filter based on mask area ratio
        if mask_box_ratio < min_mask_ratio:
            print(f"Skipping detection {i} with mask/box ratio {mask_box_ratio:.3f} (mask pixels: {mask_area_pixels}, box pixels: {box_area_pixels})")
            continue
        
        keep_indices.append(i)
    
    return torch.tensor(keep_indices)

def apply_nms(boxes, scores, masks, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to boxes and associated masks.
    
    Args:
        boxes: Tensor of bounding boxes (N, 4)
        scores: Tensor of confidence scores (N)
        masks: List of binary masks
        iou_threshold: IoU threshold for NMS
    
    Returns:
        keep_indices: Indices of boxes to keep after NMS
    """
    # Convert boxes to tensor if not already
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes)
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores)
    
    # Apply NMS and return indices
    return nms(boxes, scores, iou_threshold)

def run_inference(model_path, image_paths, output_dir="predictions", 
                 conf_threshold=0.25, iou_threshold=0.5,
                 max_box_ratio=0.8, min_mask_ratio=0.01):
    """
    Run inference on specified images using the trained YOLO model.
    
    Args:
        model_path: Path to the trained .pt model
        image_paths: List of paths to images for inference
        output_dir: Directory to save visualization results
        conf_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for NMS
        max_box_ratio: Maximum allowed box area ratio
        min_mask_ratio: Minimum mask area ratio
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    model = YOLO(model_path)
    print(f"Loaded model from {model_path}")
    
    # Process each image
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"Warning: Image not found at {img_path}")
            continue
            
        print(f"\nProcessing {img_path}")
        
        # Read the image
        original_image = cv2.imread(img_path)
        if original_image is None:
            print(f"Error: Could not read image at {img_path}")
            continue
        
        # Store original size
        original_size = original_image.shape[:2]
        
        # Preprocess image to 640x640
        processed_image, scale, pad = preprocess_image(original_image, target_size=640)
        
        # Run inference
        results = model.predict(
            source=processed_image,
            conf=conf_threshold,
            save=False,
            show=False
        )
        
        # Apply filtering and NMS if we have detections
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            # Get boxes, scores and masks
            boxes = results[0].boxes.xyxy
            scores = results[0].boxes.conf
            cls = results[0].boxes.cls
            masks = results[0].masks
            
            # Get original number of detections
            orig_num = len(boxes)
            
            # First apply quality filtering
            quality_indices = filter_detections(
                boxes, scores, cls, masks,
                conf_threshold=conf_threshold,
                max_box_ratio=max_box_ratio,
                min_mask_ratio=min_mask_ratio
            )
            
            if len(quality_indices) > 0:
                # Apply NMS on filtered detections
                boxes_filtered = boxes[quality_indices]
                scores_filtered = scores[quality_indices]
                cls_filtered = cls[quality_indices]
                
                nms_indices = apply_nms(boxes_filtered, scores_filtered, masks.data, iou_threshold)
                
                # Combine indices
                final_indices = quality_indices[nms_indices]
                
                # Create new Results object with filtered detections
                new_results = Results(
                    orig_img=results[0].orig_img,
                    path=results[0].path,
                    names=results[0].names
                )
                
                # Create data in the format expected by Boxes (xyxy, conf, cls)
                filtered_boxes = torch.cat([
                    boxes[final_indices],
                    scores[final_indices].unsqueeze(1),
                    cls[final_indices].unsqueeze(1)
                ], dim=1)
                
                # Update boxes with the correctly formatted data
                new_results.boxes = Boxes(filtered_boxes, orig_shape=results[0].boxes.orig_shape)
                
                # Update masks if they exist
                if masks is not None:
                    new_results.masks = results[0].masks[final_indices]
                
                # Replace original results with filtered results
                results[0] = new_results
                
                print(f"Filtering applied: {orig_num} -> {len(quality_indices)} -> {len(final_indices)} detections")
            else:
                print("No detections passed quality filtering")
                results[0].boxes = None
                results[0].masks = None
        
        # Generate output path
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(output_dir, f"{base_name}_prediction.png")
        
        # Visualize and save results
        visualize_prediction(
            original_image,  # Use original image for visualization
            results,
            save_path,
            conf_threshold,
            original_size=original_size,
            scale=scale,
            pad=pad
        )
        
        if results[0].boxes is not None:
            num_detections = len(results[0].boxes)
            print(f"Final detections: {num_detections} objects with confidence >= {conf_threshold}")
        else:
            print("No valid detections found")

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "19June_0022.pt"
    
    # Example usage with two images from yolo_dataset
    IMAGE_PATHS = [
        "yolo_dataset/val/images/000001.jpg",  # Replace with your image paths
        "yolo_dataset/val/images/000002.jpg"
    ]
    
    # Run inference with additional filtering
    run_inference(
        MODEL_PATH, 
        IMAGE_PATHS, 
        conf_threshold=0.25,
        iou_threshold=0.3,
        max_box_ratio=0.8,  # Maximum box size as ratio of image area (80%)
        min_mask_ratio=0.1  # Minimum mask area as ratio of box area (10%)
    )

