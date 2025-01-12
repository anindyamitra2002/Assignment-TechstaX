import pandas as pd
import numpy as np
import os
import torch
import pytorch_lightning as pl
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import json
from glob import glob
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

# Import your existing classes
from utils import AccidentPredictor, AccidentDataset, LitProgressBar

class MetricsLogger:
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = self.log_dir / f"training_metrics_remaining_chunks_{self.timestamp}.txt"
        self.summary_file = self.log_dir / f"training_summary_remaining_chunks_{self.timestamp}.json"
        self.chunk_metrics = {}
        
    def log_chunk_start(self, chunk_idx: int):
        with open(self.metrics_file, 'a') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Starting training on chunk {chunk_idx}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*50}\n")
        
        self.chunk_metrics[chunk_idx] = {
            'epochs': [],
            'best_val_loss': float('inf'),
            'best_val_acc': 0,
            'best_val_f1': 0
        }
    
    def log_epoch_metrics(self, chunk_idx: int, epoch: int, metrics: dict):
        epoch_log = (
            f"Epoch {epoch:02d} - "
            f"Train Loss: {metrics['train_loss']:.4f} - "
            f"Train Acc: {metrics['train_acc']:.4f} - "
            f"Train F1: {metrics['train_f1']:.4f} - "
            f"Val Loss: {metrics['val_loss']:.4f} - "
            f"Val Acc: {metrics['val_acc']:.4f} - "
            f"Val F1: {metrics['val_f1']:.4f}\n"
        )
        
        with open(self.metrics_file, 'a') as f:
            f.write(epoch_log)
        
        self.chunk_metrics[chunk_idx]['epochs'].append(metrics)
        
        # Update best metrics
        if metrics['val_loss'] < self.chunk_metrics[chunk_idx]['best_val_loss']:
            self.chunk_metrics[chunk_idx]['best_val_loss'] = metrics['val_loss']
        if metrics['val_acc'] > self.chunk_metrics[chunk_idx]['best_val_acc']:
            self.chunk_metrics[chunk_idx]['best_val_acc'] = metrics['val_acc']
        if metrics['val_f1'] > self.chunk_metrics[chunk_idx]['best_val_f1']:
            self.chunk_metrics[chunk_idx]['best_val_f1'] = metrics['val_f1']
    
    def save_summary(self):
        summary = {
            'training_start': self.timestamp,
            'training_end': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'chunk_metrics': self.chunk_metrics
        }
        
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=4)

class MetricsCallback(pl.Callback):
    def __init__(self, metrics_logger, chunk_idx):
        self.metrics_logger = metrics_logger
        self.chunk_idx = chunk_idx
    
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = {
            'train_loss': trainer.callback_metrics['train_loss'].item(),
            'train_acc': trainer.callback_metrics['train_acc'].item(),
            'train_f1': trainer.callback_metrics['train_f1'].item(),
            'val_loss': trainer.callback_metrics['val_loss'].item(),
            'val_acc': trainer.callback_metrics['val_acc'].item(),
            'val_f1': trainer.callback_metrics['val_f1'].item()
        }
        self.metrics_logger.log_epoch_metrics(self.chunk_idx, trainer.current_epoch, metrics)

def get_remaining_chunks(data_dir: str):
    """Get all remaining chunks except chunk_0"""
    train_chunks = sorted(glob(os.path.join(data_dir, 'train_chunk_*.csv')))
    val_chunks = sorted(glob(os.path.join(data_dir, 'val_chunk_*.csv')))
    
    # Remove chunk_0 files
    train_chunks = [f for f in train_chunks if not f.endswith('_0.csv')]
    val_chunks = [f for f in val_chunks if not f.endswith('_0.csv')]
    
    return train_chunks, val_chunks

def train_remaining_chunks(
    data_dir: str,
    model_dir: str = 'models',
    log_dir: str = 'logs',
    batch_size: int = 32,
    epochs_per_chunk: int = 10,
    project_name: str = 'accident-severity-prediction'
):
    """
    Train the model on remaining chunks, starting from the saved state of chunk_0
    """
    # Get remaining chunks
    train_chunks, val_chunks = get_remaining_chunks(data_dir)
    print(f"Found {len(train_chunks)} remaining train chunks and {len(val_chunks)} remaining val chunks")
    
    # Load the model from chunk_0
    checkpoint_path = f'{model_dir}/model_after_chunk_0.pt'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Couldn't find checkpoint from chunk_0 at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    
    # Initialize metrics logger
    metrics_logger = MetricsLogger(log_dir)
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
    
    # Load first chunk to get dimensions
    temp_dataset = AccidentDataset(pd.read_csv(train_chunks[0]), tokenizer)
    
    # Initialize model with saved weights
    model = AccidentPredictor(
        tabular_dim=temp_dataset.tabular_features.shape[1],
        embedding_dim=temp_dataset.text_embeddings.shape[1],
        num_classes=len(temp_dataset.label_encoder.classes_)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    label_encoder = checkpoint['label_encoder']
    
    del temp_dataset
    
    # Train on remaining chunks
    for chunk_idx, (train_chunk, val_chunk) in enumerate(zip(train_chunks, val_chunks), start=1):
        # Get actual chunk number from filename
        actual_chunk_num = int(train_chunk.split('_')[-1].split('.')[0])
        metrics_logger.log_chunk_start(actual_chunk_num)
        print(f"\nTraining on chunk {actual_chunk_num} ({chunk_idx}/{len(train_chunks)})")
        
        # Load data chunks
        train_df = pd.read_csv(train_chunk)
        val_df = pd.read_csv(val_chunk)
        
        # Create datasets
        train_dataset = AccidentDataset(train_df, tokenizer, label_encoder=label_encoder)
        val_dataset = AccidentDataset(
            val_df, 
            tokenizer,
            text_embeddings=train_dataset.get_text_embeddings(val_df['Description'].tolist()),
            label_encoder=label_encoder
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize wandb logger
        wandb_logger = WandbLogger(
            project=project_name,
            name=f'hybrid-model-chunk-{actual_chunk_num}',
            log_model=True
        )
        
        # Initialize trainer
        trainer = pl.Trainer(
            max_epochs=epochs_per_chunk,
            logger=wandb_logger,
            callbacks=[
                ModelCheckpoint(
                    dirpath=f'{model_dir}/chunk_{actual_chunk_num}',
                    filename='accident-predictor-{epoch:02d}-{val_loss:.2f}',
                    save_top_k=1,
                    mode='min'
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    mode='min'
                ),
                LitProgressBar(),
                MetricsCallback(metrics_logger, actual_chunk_num)
            ],
            accelerator='auto',
            devices=1,
            log_every_n_steps=10
        )
        
        # Train model
        trainer.fit(model, train_loader, val_loader)
        
        # Save model after each chunk
        chunk_save_path = f'{model_dir}/model_after_chunk_{actual_chunk_num}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'label_encoder': label_encoder
        }, chunk_save_path)
        
        # Clear memory
        del train_dataset, val_dataset, train_loader, val_loader
        torch.cuda.empty_cache()
    
    # Save final model and training summary
    final_save_path = f'{model_dir}/final_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_encoder': label_encoder
    }, final_save_path)
    
    metrics_logger.save_summary()
    return model, label_encoder

if __name__ == "__main__":
    # Define paths
    data_dir = '/teamspace/studios/this_studio/Assignment-TechstaX/data/data_chunks'
    model_dir = '/teamspace/studios/this_studio/Assignment-TechstaX/models'
    log_dir = '/teamspace/studios/this_studio/Assignment-TechstaX/logs'
    
    # Train on remaining chunks
    model, label_encoder = train_remaining_chunks(
        data_dir=data_dir,
        model_dir=model_dir,
        log_dir=log_dir,
        batch_size=32,
        epochs_per_chunk=10,
        project_name='accident-severity-prediction'
    )