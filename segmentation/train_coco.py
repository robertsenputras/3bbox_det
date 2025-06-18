from ultralytics import YOLO
import os

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
    results_stage1 = model.train(
        data="yolo.yaml", 
        epochs=10, 
        imgsz=(640, 640), 
        batch=16, 
        workers=8, 
        device="mps", 
        save=True, 
        freeze=10,  # Freeze first 10 layers
        lr0=0.0005, 
        patience=50, 
        save_period=10,
        name="stage1_frozen"  # Custom name for this training run
    )
    
    # Get the path to the best model from stage 1
    stage1_best_model = results_stage1.save_dir / "weights" / "best.pt"
    print(f"Stage 1 completed. Best model saved at: {stage1_best_model}")
    
    # Stage 2: Resume training with unfrozen weights for 50 more epochs
    print("\n=== Stage 2: Resume training with unfrozen weights (50 epochs) ===")
    
    # Load the best model from stage 1
    model_stage2 = YOLO(stage1_best_model)
    
    # Train with unfrozen weights
    results_stage2 = model_stage2.train(
        data="yolo.yaml", 
        epochs=50, 
        imgsz=(640, 640), 
        batch=16, 
        workers=8, 
        device="mps", 
        save=True, 
        freeze=0,  # No freezing - all layers trainable
        lr0=0.0001,  # Lower learning rate for fine-tuning
        patience=50, 
        save_period=10,
        name="stage2_unfrozen",  # Custom name for this training run
        resume=False  # Start fresh training (not resume from checkpoint)
    )
    
    stage2_best_model = results_stage2.save_dir / "weights" / "best.pt"
    print(f"Stage 2 completed. Final best model saved at: {stage2_best_model}")
    
    return results_stage1, results_stage2

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
    results = model.train(
        data="yolo.yaml", 
        epochs=additional_epochs, 
        imgsz=(640, 640), 
        batch=16, 
        workers=8, 
        device="mps", 
        save=True, 
        freeze=0,  # Unfreeze all layers
        lr0=0.0001,  # Lower learning rate for fine-tuning
        patience=50, 
        save_period=10,
        name="resumed_unfrozen",
        resume=False  # Start new training session
    )
    
    return results

if __name__ == "__main__":
    # Option 1: Full two-stage training from scratch
    results_stage1, results_stage2 = train_with_freezing_schedule()
    
    # # Option 2: Resume from existing model (if you already have a trained model)
    # # Uncomment and modify the path below if you want to resume from existing model
    # existing_model_path = "19June_0022.pt"  # Replace with your model path
    # if os.path.exists(existing_model_path):
    #     print(f"Found existing model: {existing_model_path}")
    #     print("Choose option:")
    #     print("1. Resume from existing model with unfrozen weights")
    #     print("2. Start fresh two-stage training")
        
    #     # For automatic execution, choose option 1 (resume)
    #     choice = "1"  # Change to "2" if you want fresh training
        
    #     if choice == "1":
    #         results = resume_existing_training(existing_model_path, additional_epochs=50)
    #     else:
    #         results_stage1, results_stage2 = train_with_freezing_schedule()
    # else:
    #     print("No existing model found. Starting fresh two-stage training.")
    #     results_stage1, results_stage2 = train_with_freezing_schedule()
