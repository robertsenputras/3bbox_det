import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'frustum_pointnets_pytorch'))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import yaml
from datetime import datetime

# Import the model directly from your frustum_detection directory
from frustum_pointnets_pytorch.models.frustum_pointnets_v1 import FrustumPointNetv1
from frustum_pointnets_pytorch.train.provider_fpointnet import compute_box3d_iou
from frustum_pointnets_pytorch.models.model_util import FrustumPointNetLoss
from frustum_pointnets_pytorch.train import provider_fpointnet
from dataset import FrustumDataset

def custom_collate_fn(batch):
    """Custom collate function to handle batches with different numbers of objects
    Args:
        batch: List of dictionaries from __getitem__
    Returns:
        Collated batch with padded tensors
    """
    # Find maximum number of objects in this batch
    max_objects = max([b['point_cloud'].size(0) for b in batch])
    
    # Initialize the collated batch
    collated_batch = {}
    
    # Handle each key in the batch
    for key in batch[0].keys():
        if key == 'num_objects':
            # Just stack the number of objects
            collated_batch[key] = torch.stack([b[key] for b in batch])
            continue
            
        if key == 'one_hot':
            # One hot is already fixed size
            collated_batch[key] = torch.stack([b[key] for b in batch])
            continue
            
        if isinstance(batch[0][key], torch.Tensor):
            # Get the shape of the tensor
            tensor_shape = batch[0][key].shape
            
            # Create padded tensors for each item in batch
            if len(tensor_shape) == 1:  # For 1D tensors like rot_angle
                padded = [torch.cat([b[key], 
                                   torch.zeros(max_objects - len(b[key]), 
                                             dtype=b[key].dtype, 
                                             device=b[key].device)]) 
                         for b in batch]
            elif len(tensor_shape) == 2:  # For 2D tensors
                if key in ['size_residual', 'box3d_center']:  # Special case for Nx3 tensors
                    padded = [torch.cat([b[key], 
                                       torch.zeros(max_objects - b[key].size(0), 3,
                                                 dtype=b[key].dtype,
                                                 device=b[key].device)], 0)
                             for b in batch]
                else:  # For other 2D tensors
                    padded = [torch.cat([b[key],
                                       torch.zeros(max_objects - b[key].size(0),
                                                 b[key].size(1),
                                                 dtype=b[key].dtype,
                                                 device=b[key].device)], 0)
                             for b in batch]
            elif len(tensor_shape) == 3:  # For 3D tensors like point_cloud
                padded = [torch.cat([b[key],
                                   torch.zeros(max_objects - b[key].size(0),
                                             b[key].size(1),
                                             b[key].size(2),
                                             dtype=b[key].dtype,
                                             device=b[key].device)], 0)
                         for b in batch]
                
            # Stack the padded tensors
            collated_batch[key] = torch.stack(padded)
    
    return collated_batch

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

class FrustumLoss:
    def __init__(self, device):
        self.device = device
        self.Loss = FrustumPointNetLoss().to(device)

    def compute_loss(self, predictions, targets):
        """
        Compute all losses and metrics
        Args:
            predictions: dict containing model predictions
            targets: dict containing ground truth
        Returns:
            losses: dict of loss values
            metrics: dict of metrics
        """
        bs = predictions['logits'].shape[0]
        
        # Get actual number of objects for each batch item
        num_objects = targets['num_objects']  # (B,)
        max_objects = predictions['logits'].shape[1]
        
        # Create mask for valid objects
        valid_mask = torch.zeros((bs, max_objects), dtype=torch.bool, device=self.device)
        for i in range(bs):
            valid_mask[i, :num_objects[i]] = True
        
        # Compute losses only for valid objects
        loss, loss_dict = self.Loss(predictions, targets, valid_mask)
        
        # Normalize losses by actual number of objects
        total_objects = num_objects.sum().item()
        if total_objects > 0:
            loss = loss / total_objects
            for key in loss_dict:
                loss_dict[key] = loss_dict[key] / total_objects

        # Compute metrics
        with torch.no_grad():
            # Reshape logits and targets for accuracy computation
            valid_logits = predictions['logits'][valid_mask]
            valid_targets = targets['seg'][valid_mask]
            n_points = valid_logits.shape[-1]
            
            # Compute segmentation accuracy
            seg_correct = torch.argmax(valid_logits.detach().cpu(), 2).eq(valid_targets.detach().cpu()).numpy()
            seg_accuracy = np.sum(seg_correct) / float(seg_correct.size) if seg_correct.size > 0 else 0.0

            # Compute IoU metrics only for valid objects
            if total_objects > 0:
                iou2ds, iou3ds = compute_box3d_iou(
                    predictions['box3d_center'][valid_mask].detach().cpu().numpy(),
                    predictions['heading_scores'][valid_mask].detach().cpu().numpy(),
                    predictions['heading_residual'][valid_mask].detach().cpu().numpy(),
                    predictions['size_scores'][valid_mask].detach().cpu().numpy(),
                    predictions['size_residual'][valid_mask].detach().cpu().numpy(),
                    targets['box3d_center'][valid_mask].detach().cpu().numpy(),
                    targets['angle_class'][valid_mask].detach().cpu().numpy(),
                    targets['angle_residual'][valid_mask].detach().cpu().numpy(),
                    targets['size_class'][valid_mask].detach().cpu().numpy(),
                    targets['size_residual'][valid_mask].detach().cpu().numpy()
                )
            else:
                iou2ds = np.array([0.0])
                iou3ds = np.array([0.0])

        metrics = {
            'seg_acc': seg_accuracy,
            'iou2d': iou2ds.mean(),
            'iou3d': iou3ds.mean(),
            'iou3d_0.7': np.sum(iou3ds >= 0.7)/total_objects if total_objects > 0 else 0.0
        }

        return loss_dict, metrics

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
        
        # Initialize loss
        self.criterion = FrustumLoss(self.device)
        
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
        
        # Create datasets with simplified parameters
        train_dataset = FrustumDataset(
            data_path=self.config['data_root'],
            scene_list=train_scenes,
            num_points=self.config['num_points']
        )
        
        val_dataset = FrustumDataset(
            data_path=self.config['data_root'],
            scene_list=val_scenes,
            num_points=self.config['num_points']
        )
        
        # Create dataloaders with custom collate function
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            collate_fn=custom_collate_fn  # Use our custom collate function
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            collate_fn=custom_collate_fn  # Use our custom collate function
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
            # # print all shape
            # print("point_cloud", data_dicts['point_cloud'].shape)
            # print("one_hot", data_dicts['one_hot'].shape)
            # print("box3d_center", data_dicts['box3d_center'].shape)
            # print("seg", data_dicts['seg'].shape)
            # print("size_class", data_dicts['size_class'].shape)
            # print("size_residual", data_dicts['size_residual'].shape)
            # print("angle_class", data_dicts['angle_class'].shape)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(data_dicts)
            
            # Compute loss and metrics
            losses, metrics = self.criterion.compute_loss(predictions, data_dicts)
            
            # Total loss
            total_loss = sum(losses.values())
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            # Log metrics
            if self.config.get('log_metrics_every', 0) > 0 and batch % self.config['log_metrics_every'] == 0:
                logging.info(f"Metrics: {metrics}")
                logging.info(f"Losses: {losses}")
            
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
                predictions = self.model(data_dicts)
                
                # Compute loss and metrics
                losses, metrics = self.criterion.compute_loss(predictions, data_dicts)
                
                # Total loss
                batch_loss = sum(losses.values()).item()
                total_loss += batch_loss
                
                # Log validation metrics
                if self.config.get('log_val_metrics_every', 0) > 0 and batch % self.config['log_val_metrics_every'] == 0:
                    logging.info(f"Validation Metrics: {metrics}")
                    logging.info(f"Validation Losses: {losses}")
        
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