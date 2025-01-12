import pandas as pd
import numpy as np
import torch
import json
import os
from datetime import datetime
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import wandb

# Import your dataset class
from utils import AccidentDataset

class ModelEvaluator:
    def __init__(self, model_path: str, test_data_path: str, output_dir: str):
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
        
        # Load model and data
        self.load_model()
        self.load_test_data()
        
    def load_model(self):
        """Load the trained model and label encoder"""
        print("Loading model from:", self.model_path)
        checkpoint = torch.load(self.model_path)
        self.label_encoder = checkpoint['label_encoder']
        
        # Initialize model with same architecture
        # Note: You'll need to initialize with same parameters as training
        test_data = pd.read_csv(self.test_data_path)
        temp_dataset = AccidentDataset(test_data, self.tokenizer)
        
        from utils import AccidentPredictor
        self.model = AccidentPredictor(
            tabular_dim=temp_dataset.tabular_features.shape[1],
            embedding_dim=temp_dataset.text_embeddings.shape[1],
            num_classes=len(self.label_encoder.classes_)
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        del temp_dataset
        
    def load_test_data(self):
        """Load and prepare test data"""
        print("Loading test data from:", self.test_data_path)
        test_df = pd.read_csv(self.test_data_path)
        self.test_dataset = AccidentDataset(
            test_df, 
            self.tokenizer,
            label_encoder=self.label_encoder,
            is_test=False  # Set to False to get actual labels
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
    def evaluate(self):
        """Evaluate the model and compute all metrics"""
        print("Starting evaluation...")
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                tabular_features, text_embedding, labels = batch
                
                # Move to device
                tabular_features = tabular_features.to(self.device)
                text_embedding = text_embedding.to(self.device)
                
                # Get predictions
                outputs = self.model(tabular_features, text_embedding)
                preds = torch.argmax(outputs, dim=1)
                
                # Store predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert predictions to original labels
        self.pred_labels = self.label_encoder.inverse_transform(all_preds)
        self.true_labels = self.label_encoder.inverse_transform(all_labels)
        
        # Compute metrics
        self.compute_metrics()
        
    def compute_metrics(self):
        """Compute and store all evaluation metrics"""
        # Basic metrics
        self.metrics = {
            'accuracy': accuracy_score(self.true_labels, self.pred_labels),
            'macro_f1': f1_score(self.true_labels, self.pred_labels, average='macro'),
            'weighted_f1': f1_score(self.true_labels, self.pred_labels, average='weighted'),
            'macro_precision': precision_score(self.true_labels, self.pred_labels, average='macro'),
            'macro_recall': recall_score(self.true_labels, self.pred_labels, average='macro')
        }
        
        # Classification report
        self.class_report = classification_report(
            self.true_labels, 
            self.pred_labels,
            output_dict=True
        )
        
        # Confusion matrix
        self.conf_matrix = confusion_matrix(self.true_labels, self.pred_labels)
        
    def plot_confusion_matrix(self):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            self.conf_matrix,
            annot=True,
            fmt='d',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(self.output_dir / f'confusion_matrix_{self.timestamp}.png')
        plt.close()
        
    def save_results(self):
        """Save all evaluation results"""
        # Save metrics to JSON
        metrics_file = self.output_dir / f'evaluation_metrics_{self.timestamp}.json'
        results = {
            'basic_metrics': self.metrics,
            'classification_report': self.class_report,
            'confusion_matrix': self.conf_matrix.tolist(),
            'evaluation_timestamp': self.timestamp,
            'model_path': self.model_path,
            'test_data_path': self.test_data_path
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(results, f, indent=4)
            
        # Save detailed results to text file
        report_file = self.output_dir / f'evaluation_report_{self.timestamp}.txt'
        with open(report_file, 'w') as f:
            f.write("=== Model Evaluation Report ===\n\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Test Data: {self.test_data_path}\n\n")
            
            f.write("=== Basic Metrics ===\n")
            for metric, value in self.metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            
            f.write("\n=== Classification Report ===\n")
            f.write(classification_report(self.true_labels, self.pred_labels))
            
            f.write("\nResults have been saved to:\n")
            f.write(f"- Metrics JSON: {metrics_file}\n")
            f.write(f"- Confusion Matrix Plot: {self.output_dir}/confusion_matrix_{self.timestamp}.png\n")
    
    def log_to_wandb(self, project_name: str = "accident-severity-prediction"):
        """Log results to Weights & Biases"""
        wandb.init(project=project_name, name=f"evaluation_{self.timestamp}")
        
        # Log metrics
        wandb.log(self.metrics)
        
        # Log confusion matrix
        wandb.log({
            "confusion_matrix": wandb.plots.HeatMap(
                list(self.label_encoder.classes_),
                list(self.label_encoder.classes_),
                self.conf_matrix.tolist(),
                show_text=True
            )
        })
        
        # Log classification report
        wandb.log({"classification_report": wandb.Table(
            columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
            data=[
                [class_name, metrics['precision'], metrics['recall'], 
                 metrics['f1-score'], metrics['support']]
                for class_name, metrics in self.class_report.items()
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']
            ]
        )})
        
        wandb.finish()

def main():
    # Define paths
    model_path = '/teamspace/studios/this_studio/Assignment-TechstaX/models/final_model.pt'
    test_data_path = '/teamspace/studios/this_studio/Assignment-TechstaX/data/test_data.csv'
    output_dir = '/teamspace/studios/this_studio/Assignment-TechstaX/evaluation_reports'
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=model_path,
        test_data_path=test_data_path,
        output_dir=output_dir
    )
    
    # Run evaluation
    evaluator.evaluate()
    evaluator.plot_confusion_matrix()
    evaluator.save_results()
    evaluator.log_to_wandb()
    
    # Print summary
    print("\nEvaluation completed! Results saved to:", output_dir)
    print("\nKey Metrics:")
    for metric, value in evaluator.metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()