"""
Production GPU Training Script for Transformer-based Cryptocurrency Trading
Full-scale training pipeline optimized for high-end GPU servers
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import logging
from typing import Dict, List, Tuple, Optional
import json
import argparse
from tqdm import tqdm
import wandb  # Weights & Biases for experiment tracking
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import local modules
from gpu_transformer_model import (
    ProductionCryptoTransformer, 
    ProductionFeaturesExtractor, 
    ProductionTradingEnvironment,
    create_production_model_config
)
from gpu_data_processing import ProductionDataProcessor

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gpu_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CryptoSequenceDataset(Dataset):
    """
    Dataset class for sequence-based cryptocurrency data
    """
    def __init__(
        self, 
        data: pd.DataFrame, 
        sequence_length: int = 200,
        prediction_horizon: int = 1,
        assets: List[str] = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL']
    ):
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.assets = assets
        
        # Identify feature columns (exclude timestamp and target columns)
        self.feature_cols = [col for col in data.columns if col != 'timestamp']
        self.target_cols = [f'{asset}_close' for asset in assets if f'{asset}_close' in data.columns]
        
        # Create sequences
        self.sequences = []
        self.targets = []
        
        logger.info(f"Creating sequences with length {sequence_length}")
        
        for i in range(len(data) - sequence_length - prediction_horizon + 1):
            # Input sequence
            seq_data = data.iloc[i:i + sequence_length][self.feature_cols].values
            
            # Target (future prices)
            target_data = data.iloc[i + sequence_length + prediction_horizon - 1][self.target_cols].values
            
            if not (np.isnan(seq_data).any() or np.isnan(target_data).any()):
                self.sequences.append(seq_data.astype(np.float32))
                self.targets.append(target_data.astype(np.float32))
        
        logger.info(f"Created {len(self.sequences)} sequences")
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.targets[idx])

class ProductionTrainer:
    """
    Production training pipeline for transformer models
    """
    
    def __init__(
        self,
        model_config: Dict,
        training_config: Dict,
        data_path: str,
        experiment_name: str = "crypto_transformer_production"
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.data_path = data_path
        self.experiment_name = experiment_name
        
        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            logger.info("Using Apple Silicon GPU (MPS)")
        else:
            self.device = torch.device('cpu')
            logger.warning("Using CPU - training will be slow!")
        
        # Initialize wandb for experiment tracking
        wandb.init(
            project="crypto-transformer-production",
            name=experiment_name,
            config={**model_config, **training_config}
        )
        
        self.setup_model()
        
    def setup_model(self):
        """Initialize model, optimizer, and loss function"""
        # Load data to determine input dimensions
        logger.info("Loading training data...")
        data = pd.read_csv(self.data_path)
        
        # Determine input dimensions
        feature_cols = [col for col in data.columns if col != 'timestamp']
        input_dim = len(feature_cols)
        
        logger.info(f"Input dimension: {input_dim}")
        
        # Initialize model
        self.model = ProductionCryptoTransformer(
            input_dim=input_dim,
            **self.model_config['model_params']
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Model initialized with {total_params:,} total parameters")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Setup optimizer with advanced features
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.training_config['learning_rate'],
            weight_decay=self.training_config['weight_decay'],
            betas=(0.9, 0.95)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.training_config['learning_rate'],
            epochs=self.training_config['n_epochs'],
            steps_per_epoch=100,  # Will be updated after data loading
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Multiple loss functions for different heads
        self.price_loss_fn = nn.MSELoss()
        self.regime_loss_fn = nn.CrossEntropyLoss()
        self.confidence_loss_fn = nn.BCELoss()
        
        # For gradient clipping
        self.max_grad_norm = self.training_config.get('gradient_clip', 1.0)
        
    def create_data_loaders(self, test_size: float = 0.2):
        """Create train and validation data loaders"""
        logger.info("Creating data loaders...")
        
        # Load data
        data = pd.read_csv(self.data_path)
        logger.info(f"Loaded data with shape: {data.shape}")
        
        # Split data temporally (not randomly for time series)
        split_idx = int(len(data) * (1 - test_size))
        train_data = data.iloc[:split_idx]
        val_data = data.iloc[split_idx:]
        
        # Create datasets
        assets = self.model_config['model_params'].get('assets', ['BTC', 'ETH', 'BNB', 'ADA', 'SOL'])
        sequence_length = self.model_config['model_params']['max_seq_len']
        
        self.train_dataset = CryptoSequenceDataset(
            train_data, 
            sequence_length=sequence_length,
            assets=assets
        )
        
        self.val_dataset = CryptoSequenceDataset(
            val_data, 
            sequence_length=sequence_length,
            assets=assets
        )
        
        # Create data loaders
        batch_size = self.training_config['batch_size']
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Update scheduler steps per epoch
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.training_config['learning_rate'],
            epochs=self.training_config['n_epochs'],
            steps_per_epoch=len(self.train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        logger.info(f"Train dataset: {len(self.train_dataset)} samples")
        logger.info(f"Validation dataset: {len(self.val_dataset)} samples")
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Validation batches: {len(self.val_loader)}")
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        price_loss_total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (sequences, targets) in enumerate(progress_bar):
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            
            # Calculate losses for different heads
            price_loss = self.price_loss_fn(outputs['price_predictions'][:, :, 0], targets)
            
            # Portfolio optimization loss
            portfolio_loss = torch.mean(torch.sum(outputs['portfolio_weights'], dim=1) - 1.0) ** 2
            
            # Confidence calibration loss
            confidence_target = torch.ones(outputs['confidence'].size(0), 1).to(self.device) * 0.7
            confidence_loss = self.confidence_loss_fn(outputs['confidence'], confidence_target)
            
            # Combined loss
            total_batch_loss = (
                price_loss + 
                0.1 * portfolio_loss + 
                0.05 * confidence_loss
            )
            
            # Backward pass
            total_batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += total_batch_loss.item()
            price_loss_total += price_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_batch_loss.item():.6f}',
                'Price': f'{price_loss.item():.6f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log to wandb every 100 batches
            if batch_idx % 100 == 0:
                wandb.log({
                    'batch_loss': total_batch_loss.item(),
                    'price_loss': price_loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch
                })
            
            # Clear cache periodically for GPU
            if torch.cuda.is_available() and batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(self.train_loader)
        avg_price_loss = price_loss_total / len(self.train_loader)
        
        return {
            'total_loss': avg_loss,
            'price_loss': avg_price_loss
        }
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        price_loss_total = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f"Validation {epoch+1}")
            
            for sequences, targets in progress_bar:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(sequences)
                
                # Calculate losses
                price_loss = self.price_loss_fn(outputs['price_predictions'][:, :, 0], targets)
                portfolio_loss = torch.mean(torch.sum(outputs['portfolio_weights'], dim=1) - 1.0) ** 2
                confidence_target = torch.ones(outputs['confidence'].size(0), 1).to(self.device) * 0.7
                confidence_loss = self.confidence_loss_fn(outputs['confidence'], confidence_target)
                
                total_batch_loss = price_loss + 0.1 * portfolio_loss + 0.05 * confidence_loss
                
                total_loss += total_batch_loss.item()
                price_loss_total += price_loss.item()
                
                # Store predictions for metrics
                predictions = outputs['price_predictions'][:, :, 0].cpu().numpy()
                targets_np = targets.cpu().numpy()
                
                all_predictions.extend(predictions)
                all_targets.extend(targets_np)
                
                progress_bar.set_postfix({'Val Loss': f'{total_batch_loss.item():.6f}'})
        
        avg_loss = total_loss / len(self.val_loader)
        avg_price_loss = price_loss_total / len(self.val_loader)
        
        # Calculate additional metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        mse = mean_squared_error(all_targets, all_predictions)
        mae = mean_absolute_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)
        
        return {
            'total_loss': avg_loss,
            'price_loss': avg_price_loss,
            'mse': mse,
            'mae': mae,
            'r2_score': r2
        }
    
    def save_model(self, epoch: int, val_metrics: Dict, save_path: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_metrics': val_metrics,
            'model_config': self.model_config,
            'training_config': self.training_config
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")
        
        # Save to wandb
        wandb.save(save_path)
    
    def train(self):
        """Main training loop"""
        logger.info("Starting production training...")
        
        # Create data loaders
        self.create_data_loaders()
        
        # Training history
        train_history = []
        val_history = []
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        # Create checkpoints directory
        checkpoint_dir = f"checkpoints_{self.experiment_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(self.training_config['n_epochs']):
            logger.info(f"\nEpoch {epoch+1}/{self.training_config['n_epochs']}")
            
            # Training phase
            train_metrics = self.train_epoch(epoch)
            train_history.append(train_metrics)
            
            # Validation phase
            val_metrics = self.validate_epoch(epoch)
            val_history.append(val_metrics)
            
            # Log to wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': train_metrics['total_loss'],
                'val_loss': val_metrics['total_loss'],
                'val_mse': val_metrics['mse'],
                'val_mae': val_metrics['mae'],
                'val_r2': val_metrics['r2_score']
            })
            
            # Early stopping check
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                patience_counter = 0
                
                # Save best model
                best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
                self.save_model(epoch, val_metrics, best_model_path)
                
            else:
                patience_counter += 1
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
                self.save_model(epoch, val_metrics, checkpoint_path)
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            logger.info(f"Train Loss: {train_metrics['total_loss']:.6f}, Val Loss: {val_metrics['total_loss']:.6f}")
            logger.info(f"Val MSE: {val_metrics['mse']:.6f}, Val R²: {val_metrics['r2_score']:.4f}")
        
        # Save training history
        history_path = os.path.join(checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump({
                'train_history': train_history,
                'val_history': val_history
            }, f, indent=2)
        
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.6f}")
        
        # Finish wandb run
        wandb.finish()
        
        return train_history, val_history

def main():
    parser = argparse.ArgumentParser(description='Production Crypto Transformer Training')
    parser.add_argument('--data-path', type=str, required=True, help='Path to training data CSV')
    parser.add_argument('--experiment-name', type=str, default='crypto_transformer_gpu', help='Experiment name')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='Learning rate')
    
    args = parser.parse_args()
    
    # Load production configuration
    config = create_production_model_config()
    
    # Update with command line arguments
    config['training_params']['n_epochs'] = args.epochs
    config['training_params']['batch_size'] = args.batch_size
    config['training_params']['learning_rate'] = args.learning_rate
    
    # Initialize trainer
    trainer = ProductionTrainer(
        model_config=config,
        training_config=config['training_params'],
        data_path=args.data_path,
        experiment_name=args.experiment_name
    )
    
    # Start training
    train_history, val_history = trainer.train()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Experiment: {args.experiment_name}")
    print(f"Data: {args.data_path}")
    print(f"Epochs completed: {len(train_history)}")
    print("="*60)

if __name__ == "__main__":
    main()