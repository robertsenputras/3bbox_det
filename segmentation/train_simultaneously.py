from ultralytics import YOLO
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train_with_freezing_schedule():
    """
    Train model with a two-stage approach:
    1. First 10 epochs with frozen backbone
    2. Resume for 50 more epochs with unfrozen weights
    """
    
    # Stage 1: Train with frozen backbone for 10 epochs
    print("=== Stage 1: Training with frozen backbone (10 epochs) ===")
    model = YOLO("yolo11m-seg.pt")
    
    # Train with frozen backbone
    model.train(
        data="yolo.yaml", 
        epochs=30, 
        imgsz=(640, 640), 
        batch=16, 
        workers=8, 
        device=0,
        save=True, 
        freeze=10,  # Freeze first 10 layers
        lr0=0.0005, 
        patience=15,  # Stop if no improvement for 15 epochs
        save_period=10,
        name="stage1_frozen"  # Custom name for this training run
    )
    
    # Get the path to the best model from stage 1
    stage1_best_model = "runs/segment/stage1_frozen/weights/best.pt"
    print(f"Stage 1 completed. Best model saved at: {stage1_best_model}")
    
    # Stage 2: Resume training with unfrozen weights for 50 more epochs
    print("\n=== Stage 2: Resume training with unfrozen weights (50 epochs) ===")
    
    # Load the best model from stage 1
    model_stage2 = YOLO(stage1_best_model)
    
    # Train with unfrozen weights
    model_stage2.train(
        data="yolo.yaml", 
        epochs=50, 
        imgsz=(640, 640), 
        batch=16, 
        workers=8, 
        device=0,
        save=True, 
        freeze=0,  # No freezing - all layers trainable
        lr0=0.0001,  # Lower learning rate for fine-tuning
        patience=10,  # Stop if no improvement for 10 epochs
        save_period=10,
        name="stage2_unfrozen",  # Custom name for this training run
        resume=False  # Start a new training using the stage1 weights
    )
    
    stage2_best_model = "runs/segment/stage2_unfrozen/weights/best.pt"
    print(f"Stage 2 completed. Final best model saved at: {stage2_best_model}")

def resume_existing_training(checkpoint_path, additional_epochs=50):
    """
    Resume training from an existing checkpoint and unfreeze weights.
    
    Args:
        checkpoint_path: Path to the existing model checkpoint
        additional_epochs: Number of additional epochs to train
    """
    print(f"=== Resuming training from {checkpoint_path} with unfrozen weights ===")
    
    # Load existing model
    model = YOLO(checkpoint_path)
    
    # Resume training with unfrozen weights
    model.train(
        data="yolo.yaml", 
        epochs=additional_epochs, 
        imgsz=(640, 640), 
        batch=16, 
        workers=8, 
        device=0,  # Changed back to GPU
        save=True, 
        freeze=0,  # Unfreeze all layers
        lr0=0.0001,  # Lower learning rate for fine-tuning
        patience=50, 
        save_period=10,
        name="resumed_unfrozen",
        resume=False  # Start new training session
    )
    
    final_model = "runs/segment/resumed_unfrozen/weights/best.pt"
    print(f"Training completed. Final model saved at: {final_model}")
    return final_model

def train_single_stage():
    """
    Train model with dynamic freezing in a single stage:
    First part of epochs with frozen backbone, then automatically unfreeze
    """
    print("=== Starting single-stage training with dynamic freezing ===")
    model = YOLO("yolo11m-seg.pt")
    
    # Train with dynamic freezing
    model.train(
        data="yolo.yaml", 
        epochs=80,  # Total epochs
        imgsz=(640, 640), 
        batch=16, 
        workers=8, 
        device=0,
        save=True, 
        freeze=10,  # Will automatically unfreeze after 10 epochs
        lr0=0.0005, 
        patience=15,  # Stop if no improvement for 15 epochs
        save_period=10,
        name="single_stage"  # Custom name for this training run
    )
    
    best_model = "runs/segment/single_stage/weights/best.pt"
    print(f"Training completed. Best model saved at: {best_model}")

if __name__ == "__main__":
    # Single stage training
    train_single_stage()
    
    # # Uncomment below if you want to resume from a specific checkpoint
    # existing_model_path = "path_to_your_model.pt"
    # if os.path.exists(existing_model_path):
    #     resume_existing_training(existing_model_path, additional_epochs=50)
