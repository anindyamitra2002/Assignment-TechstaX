import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset, DataLoader
import wandb
from sklearn.metrics import classification_report, accuracy_score, f1_score
from typing import Dict, Tuple
from tqdm.auto import tqdm

class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description('Validation')
        return bar
    
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.set_description('Training')
        return bar
    
    
class AccidentPredictor(pl.LightningModule):
    def __init__(self, 
                 tabular_dim: int,
                 embedding_dim: int,
                 num_classes: int,
                 learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # Network architecture
        self.tabular_network = nn.Sequential(
            nn.Linear(tabular_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.text_network = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, tabular_features, text_embedding):
        tabular_features = self.tabular_network(tabular_features)
        text_features = self.text_network(text_embedding)
        combined = torch.cat([tabular_features, text_features], dim=1)
        return self.classifier(combined)
    
    def training_step(self, batch, batch_idx):
        tabular_features, text_embedding, labels = batch
        logits = self(tabular_features, text_embedding)
        loss = self.criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(labels.cpu(), preds.cpu())
        f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        self.log('train_f1', f1, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        tabular_features, text_embedding, labels = batch
        logits = self(tabular_features, text_embedding)
        loss = self.criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(labels.cpu(), preds.cpu())
        f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                   lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
        

# Modified AccidentDataset class
class AccidentDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, text_embeddings=None, 
                 is_test: bool = False, label_encoder=None):
        self.df = df
        self.tokenizer = tokenizer
        self.is_test = is_test
        
        # Process tabular features
        print("Processing tabular features...")
        self.process_tabular_features()
        
        # Store or compute text embeddings
        if text_embeddings is not None:
            self.text_embeddings = text_embeddings
        else:
            print("Computing text embeddings...")
            self.text_embeddings = self.get_text_embeddings(df['Description'].tolist())
        
        if not is_test:
            if label_encoder is None:
                self.label_encoder = LabelEncoder()
                self.labels = self.label_encoder.fit_transform(df['Severity'])
            else:
                self.label_encoder = label_encoder
                self.labels = self.label_encoder.transform(df['Severity'])
    
    def process_tabular_features(self):
        # Select all columns except 'Severity' and 'Description'
        tabular_cols = [col for col in self.df.columns 
                       if col not in ['Severity', 'Description']]
        
        # Convert boolean columns to int
        bool_cols = self.df[tabular_cols].select_dtypes(include=['bool']).columns
        for col in bool_cols:
            self.df[col] = self.df[col].astype(int)
        
        # Convert categorical columns to numeric using label encoding
        cat_cols = self.df[tabular_cols].select_dtypes(include=['object']).columns
        self.label_encoders = {}
        for col in tqdm(cat_cols, desc="Encoding categorical columns"):
            self.label_encoders[col] = LabelEncoder()
            self.df[col] = self.label_encoders[col].fit_transform(self.df[col])
        
        # Scale numeric features
        self.scaler = StandardScaler()
        self.tabular_features = self.scaler.fit_transform(self.df[tabular_cols])
    
    @staticmethod
    def get_batch_embeddings(texts: list, tokenizer, model, device='cuda', batch_size=32) -> np.ndarray:
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
            batch_texts = texts[i:i + batch_size]
            encoded_input = tokenizer(batch_texts, 
                                    padding=True, 
                                    truncation=True, 
                                    max_length=512,
                                    return_tensors='pt')
            
            # Move input to device
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            
            with torch.no_grad():
                model_output = model(**encoded_input)
                batch_embeddings = model_output[0][:, 0]  # CLS token embedding
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def get_text_embeddings(self, texts: list) -> np.ndarray:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5')
        model = model.to(device)
        model.eval()
        
        return self.get_batch_embeddings(texts, self.tokenizer, model, device)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        tabular_features = torch.FloatTensor(self.tabular_features[idx])
        text_embedding = torch.FloatTensor(self.text_embeddings[idx])
        
        if self.is_test:
            return tabular_features, text_embedding
        
        label = torch.LongTensor([self.labels[idx]])[0]
        return tabular_features, text_embedding, label

