# Accident Predictor Model Documentation

## Overview

This repository contains a PyTorch-based model for predicting accident severity using both tabular and textual data. The model leverages a combination of neural networks to process tabular features and text embeddings, ultimately classifying the severity of accidents. Below is the evaluation report on the test data:

### === Model Evaluation Report ===
| Metric                | Value   |
|-----------------------|---------|
| Timestamp             | 20250112_041331 |
| Model Path            | /models/final_model.pt |
| Test Data             | /data/test_data.csv |
| Accuracy              | 0.9380  |
| Macro F1 Score        | 0.7353  |
| Weighted F1 Score     | 0.9362  |
| Macro Precision       | 0.8184  |
| Macro Recall          | 0.6906  |

### === Classification Report ===
```
              precision    recall  f1-score   support
           1       0.84      0.41      0.55      8816
           2       0.96      0.97      0.96    653712
           3       0.83      0.85      0.84     91489
           4       0.64      0.53      0.58     18822
    accuracy                           0.94    772839
   macro avg       0.82      0.69      0.74    772839
weighted avg       0.94      0.94      0.94    772839
```

Results have been saved to:
- Metrics JSON: `/evaluation_reports/evaluation_metrics_20250112_041331.json`
- Confusion Matrix Plot: `/evaluation_reports/confusion_matrix_20250112_041331.png`

---

## Table of Contents

1. [Installation](#installation)
2. [Data Preprocessing](#data-preprocessing)
    - [Handling Missing Values](#handling-missing-values)
    - [Tabular Feature Processing](#tabular-feature-processing)
    - [Text Embeddings](#text-embeddings)
    - [Data Cleaning](#data-cleaning)
3. [Model Architecture](#model-architecture)
4. [Training and Evaluation](#training-and-evaluation)
    - [Training Pipeline](#training-pipeline)
    - [Metrics Logging](#metrics-logging)
    - [Incremental Training on Data Chunks](#incremental-training-on-data-chunks)
5. [Conclusion](#conclusion)

---

## Installation

To run this code, you need to install the following dependencies:

```bash
pip install pandas numpy scikit-learn torch pytorch-lightning transformers wandb tqdm
```

---

## Data Preprocessing

### Handling Missing Values

Before feeding the data into the model, it is crucial to handle any missing values in both numerical and categorical columns. This ensures that the model receives complete and meaningful input data.

#### Numerical Imputation

For numerical columns, we use the `IterativeImputer` from Scikit-Learn, which iteratively imputes missing values based on other available features. This method is particularly useful for datasets with complex relationships between features.

#### Categorical Imputation

For categorical columns, we use the `SimpleImputer` with the `most_frequent` strategy, which replaces missing values with the most frequent value in each column.

### Tabular Feature Processing

The `AccidentDataset` class processes tabular features by:
1. Selecting all relevant columns except `'Severity'` and `'Description'`.
2. Converting boolean columns to integers (0 or 1).
3. Encoding categorical columns using `LabelEncoder`.
4. Scaling numerical features using `StandardScaler`.

### Text Embeddings

Textual data is processed by converting each description into a fixed-size embedding using a pre-trained transformer model (`BAAI/bge-small-en-v1.5`). The embeddings are computed in batches to handle large datasets efficiently.

### Data Cleaning

Data cleaning is a critical step to ensure the quality and relevance of the dataset. We identify and remove redundant or unnecessary columns and split the dataset based on the presence of null values.

#### Redundant Columns

The following columns are considered redundant or potentially unnecessary for our analysis and are removed:

1. **Location-Related Redundancies**:
   - `End_Lat` and `End_Lng`: These are often null and the start coordinates (`Start_Lat`, `Start_Lng`) are sufficient for most analyses.
   - `Number` and `Street`: Too granular and already represented by higher-level location data.
   - `Country`: Always "US" (redundant since this is US accident data).
   - `Airport_Code`: Redundant with location coordinates and only used as a reference for weather station.
   - `Turning_Loop`: Always false in the given data.

2. **Time-Related Redundancies**:
   - `Civil_Twilight`, `Nautical_Twilight`, `Astronomical_Twilight`: Highly correlated with `Sunrise_Sunset` and provide similar information.
   - `Weather_Timestamp`: Likely redundant with `Start_Time` unless there's a significant time difference.

3. **Identification/Description**:
   - `ID`: Not needed for modeling (unique identifier).

4. **Weather-Related Redundancies**:
   - `Wind_Chill(F)`: Can be derived from Temperature and Wind_Speed.
   - Either `Temperature(F)` or `Wind_Chill(F)` could be redundant depending on your modeling needs.

5. **Address Components**:
   - `Side` (Right/Left): Likely not significant for predicting accident severity.
   - `Zipcode`: Redundant with City/County/State information.


#### Splitting Criteria

The dataset was split into three subsets based on the presence of null values:

- **Training Set**: 75% of the data, including both non-null and partial-null rows.
- **Validation Set**: 15% of the data, containing only rows with no null values.
- **Test Set**: 10% of the data, containing only rows with no null values.

This approach ensures that the training set can benefit from more data while the validation and test sets provide a clean evaluation environment.

```python
from sklearn.model_selection import train_test_split

# Split the dataset into training, validation, and test sets
train_df, temp_df = train_test_split(df, test_size=0.25, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.4, random_state=42)

# Further split the validation and test sets to ensure they contain no null values
val_df = val_df.dropna()
test_df = test_df.dropna()
```

---

## Model Architecture

The `AccidentPredictor` class defines a neural network architecture that combines tabular features and text embeddings for classification. The architecture consists of two sub-networks:

1. **Tabular Network**: Processes tabular features through a series of fully connected layers followed by ReLU activations and dropout.
2. **Text Network**: Processes text embeddings through a similar set of layers.
3. **Classifier**: Combines the outputs from both networks and passes them through additional layers to produce the final classification.

In words, the architecture can be described as follows:
- **Input Layer**: Takes in tabular features and text embeddings.
- **Tabular Network**: A series of fully connected layers that process the tabular features, transforming them into a lower-dimensional representation.
- **Text Network**: A series of fully connected layers that process the text embeddings, transforming them into a lower-dimensional representation.
- **Combination Layer**: Concatenates the outputs from the tabular and text networks.
- **Classifier**: Further processes the combined output through additional fully connected layers to produce the final classification.

---

## Training and Evaluation

### Training Pipeline

The training pipeline is designed to handle large datasets by splitting them into smaller chunks and training the model incrementally. Each chunk is trained separately, and the model's state is saved after each chunk to ensure continuity.

#### Metrics Logging

A `MetricsLogger` class is used to log metrics such as loss, accuracy, and F1-score during training. These logs are stored in a dedicated directory for easy access and analysis.

#### Incremental Training on Data Chunks

The `train_remaining_chunks` function orchestrates the training process across multiple data chunks. It loads the model's state from the previous chunk and continues training on the next chunk. This approach ensures that the model benefits from the knowledge gained in previous chunks while adapting to new data.

Steps:
1. **Load Remaining Chunks**: Identify and load the remaining data chunks for training and validation.
2. **Initialize Metrics Logger**: Create a `MetricsLogger` instance to track and log training metrics.
3. **Load Model State**: Load the model's state from the previous chunk.
4. **Process Data**: For each chunk, load the data, create datasets and dataloaders, and initialize the trainer.
5. **Train Model**: Train the model on the current chunk, logging metrics and saving checkpoints.
6. **Save Model**: After training on each chunk, save the model's state for future use.
7. **Finalize Training**: Save the final model and training summary after processing all chunks.

Example:
```python
def train_remaining_chunks(
    data_dir: str,
    model_dir: str = 'models',
    log_dir: str = 'logs',
    batch_size: int = 32,
    epochs_per_chunk: int = 10,
    project_name: str = 'accident-severity-prediction'
):
    # Get remaining chunks
    train_chunks, val_chunks = get_remaining_chunks(data_dir)
    
    # Load the model from chunk_0
    checkpoint_path = f'{model_dir}/model_after_chunk_0.pt'
    checkpoint = torch.load(checkpoint_path)
    
    # Initialize metrics logger
    metrics_logger = MetricsLogger(log_dir)
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
    
    # Load first chunk to get dimensions
    temp_dataset = AccidentDataset(pd.read_csv(train_chunks[0]), tokenizer)
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
        actual_chunk_num = int(train_chunk.split('_')[-1].split('.')[0])
        metrics_logger.log_chunk_start(actual_chunk_num)
        
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
```

### Metrics Logging

The `MetricsLogger` class logs training metrics for each epoch and each data chunk. It also saves a summary of the entire training process, including the best metrics achieved for each chunk.

Example:
```python
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
```

### Incremental Training on Data Chunks

The `train_remaining_chunks` function manages the incremental training process across multiple data chunks. It ensures that the model is trained on each chunk sequentially, with the state being saved after each chunk for future use.

---

## Conclusion

This project provides a comprehensive solution for predicting accident severity using a combination of tabular and textual data. The provided documentation covers the dataset creation, model architecture, and training/evaluation procedures, making it easy to understand and extend the work. The detailed explanation of data preprocessing, including handling missing values and feature processing, ensures that the data is properly prepared for the model. The training pipeline is designed to handle large datasets by splitting them into chunks and training the model incrementally, ensuring efficient and effective training.