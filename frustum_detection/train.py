import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import logging
import yaml
from datetime import datetime

# Import the model directly from your frustum_detection directory
from frustum_pointnets_pytorch.models.frustum_pointnets_v1 import FrustumPointNetv1
from dataset import FrustumDataset

def setup_logging(output_dir):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )

class Trainer:
    def __init__(self, config):
        """
        Initialize trainer
        Args:
            config: Dictionary containing training configuration
        """
        self.config = config
        self.device = torch.device(config['device'])
        
        # Setup output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(self.output_dir)
        
        # Save config
        with open(self.output_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)
        
        # Initialize model
        self.model = self._init_model()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['lr_step_size'],
            gamma=config['lr_gamma']
        )
        
        # Initialize dataloaders
        self.train_loader, self.val_loader = self._init_dataloaders()
        
        # Initialize best validation metrics
        self.best_val_loss = float('inf')
        
    def _init_model(self):
        """Initialize Frustum-PointNet model"""
        if self.config['model_version'] != 'v1':
            raise ValueError("Only v1 model is currently supported")
            
        model = FrustumPointNetv1()
            
        if self.config['pretrained_weights']:
            # Load weights with map_location to ensure they're on the right device
            weights = torch.load(self.config['pretrained_weights'], map_location=self.device)
            model.load_state_dict(weights)
        
        # Move model to device
        model = model.to(self.device)
        
        # Ensure Loss module is also on the correct device
        if hasattr(model, 'Loss'):
            model.Loss = model.Loss.to(self.device)
        
        return model
    
    def _init_dataloaders(self):
        """Initialize train and validation dataloaders"""
        # Get list of all data directories
        data_root = Path(self.config['data_root'])
        all_scenes = sorted([d for d in data_root.iterdir() if d.is_dir()])
        
        # Split into train and validation
        num_scenes = len(all_scenes)
        num_train = int(num_scenes * self.config['train_val_split'])
        
        # Randomly shuffle scenes
        np.random.seed(42)  # For reproducibility
        scene_indices = np.random.permutation(num_scenes)
        train_indices = scene_indices[:num_train]
        val_indices = scene_indices[num_train:]
        
        train_scenes = [all_scenes[i] for i in train_indices]
        val_scenes = [all_scenes[i] for i in val_indices]
        
        logging.info(f"Number of training scenes: {len(train_scenes)}")
        logging.info(f"Number of validation scenes: {len(val_scenes)}")
        
        # Create datasets
        train_dataset = FrustumDataset(
            data_path=self.config['data_root'],
            scene_list=train_scenes,
            rotate_to_center=self.config['rotate_to_center'],
            random_flip=self.config['random_flip'],
            random_shift=self.config['random_shift'],
            num_points=self.config['num_points']
        )
        
        val_dataset = FrustumDataset(
            data_path=self.config['data_root'],
            scene_list=val_scenes,
            rotate_to_center=self.config['rotate_to_center'],
            random_flip=False,
            random_shift=False,
            num_points=self.config['num_points']
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        for batch in tqdm(self.train_loader, desc='Training'):
            # Move all data to device
            data_dicts = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Ensure model is on correct device
            self.model = self.model.to(self.device)
            losses, metrics = self.model(data_dicts)
            
            # Total loss
            total_loss = sum(losses.values())
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            total_loss += total_loss.item()
            
        return total_loss / num_batches
    
    def validate(self):
        """Run validation"""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move all data to device
                data_dicts = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                
                # Forward pass
                losses, metrics = self.model(data_dicts)
                
                # Total loss
                total_loss += sum(losses.values()).item()
        
        return total_loss / num_batches
    
    def train(self):
        """Main training loop"""
        logging.info("Starting training...")
        
        for epoch in range(self.config['num_epochs']):
            logging.info(f"Epoch {epoch+1}/{self.config['num_epochs']}")
            
            # Train epoch
            train_loss = self.train_epoch()
            logging.info(f"Training loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate()
            logging.info(f"Validation loss: {val_loss:.4f}")
            
            # Update learning rate
            self.scheduler.step()
            
            # Save checkpoint if validation loss improved
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'config': self.config
                }
                torch.save(checkpoint, self.output_dir / 'best_model.pth')
                logging.info("Saved new best model")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to training config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add timestamp to output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config['output_dir'] = str(Path(config['output_dir']) / timestamp)
    
    # Initialize trainer and start training
    trainer = Trainer(config)
    trainer.train() 