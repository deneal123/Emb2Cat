import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from transformers import AutoModel, AutoTokenizer
import json
from datetime import datetime

from src import project_path
from src.modelling.classification_head import TextClassificationModel
from src.utils.custom_logging import setup_logging

log = setup_logging()


@dataclass
class KNNEmbed2Cat:
    path_to_data: str = "./data"
    path_to_weights: str = "./weights"
    path_to_plots: str = "./plots/knn"
    path_to_submission: str = "./submissions"
    name_model: str = "cat"
    model_path_id: str = "tabularisai/multilingual-sentiment-analysis"
    use_device: str = "cuda"
    num_classes: int = 5
    batch_size: int = 64
    n_neighbors: int = 5  # Number of neighbors for KNN
    use_model_weights: bool = True  # Whether to use pretrained weights

    def __post_init__(self):
        self.date = datetime.now()
        self.path_to_data = Path(os.path.join(project_path, self.path_to_data))
        self.path_to_weights = Path(os.path.join(project_path, self.path_to_weights, 'embed2cat'))
        self.path_to_plots = Path(os.path.join(project_path, self.path_to_plots))
        self.path_to_submission = Path(os.path.join(project_path, self.path_to_submission))
        
        self.model = None
        self.tokenizer = None
        self.idx2cat = None
        self.knn = None
        
        # Set device
        if not self.use_device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.use_device == "cpu":
            self.device = torch.device("cpu")
        elif self.use_device == "cuda":
            self.device = torch.device("cuda")
            
        # Create directories if they don't exist
        os.makedirs(self.path_to_plots, exist_ok=True)
        os.makedirs(self.path_to_submission, exist_ok=True)
        
        # Load category mapping
        self.load_category_mapping()
    
    def load_category_mapping(self):
        """Load the category mapping from cat2idx.json"""
        cat2idx_path = os.path.join(self.path_to_data, "cat2idx.json")
        log.info(f"Loading category mapping from {cat2idx_path}")
        
        try:
            with open(cat2idx_path, 'r') as f:
                cat2idx = json.load(f)
                
            # Create inverse mapping (idx2cat)
            self.idx2cat = {int(k): v for k, v in cat2idx.items()}
            self.cat2idx = {v: int(k) for k, v in cat2idx.items()}
            log.info(f"Category mapping loaded: {self.idx2cat}")
        except Exception as e:
            log.error(f"Error loading category mapping: {e}")
            raise
    
    def load_model(self):
        """Load the model and optionally its weights"""
        log.info(f"Loading model from {self.model_path_id}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path_id)
        
        # Create model architecture
        base_model = AutoModel.from_pretrained(self.model_path_id)    
        self.model = TextClassificationModel(base_model, self.num_classes)
        
        # Load model weights if specified
        if self.use_model_weights:
            model_path = os.path.join(self.path_to_weights, f"{self.name_model}.pt")
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                log.info("Model weights loaded successfully from weights folder")
            else:
                log.error(f"Model file not found at {model_path}")
                raise FileNotFoundError(f"Model file not found at {model_path}")
        else:
            log.info("Using model without pretrained weights (encoder only)")
            
        self.model.to(self.device)
        self.model.eval()

    def get_embeddings(self, texts):
        """Extract embeddings from the model for a batch of texts"""
        with torch.no_grad():
            # Tokenize texts
            encoded_inputs = self.tokenizer(
                [str(text) for text in texts], 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}
            
            # Get outputs from base model
            outputs = self.model.base_model(**inputs)
            # Get CLS token representation (same as used for classification)
            embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
            
            # Get predictions from the original model for comparison
            logits = self.model(inputs)
            _, predictions = torch.max(logits, 1)
            
            return embeddings, predictions.cpu().numpy()
    
    def load_data(self, file_name):
        """Load data from CSV file"""
        file_path = os.path.join(self.path_to_data, file_name)
        log.info(f"Loading data from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            log.info(f"Loaded {len(df)} samples")
            return df
        except Exception as e:
            log.error(f"Error loading data: {e}")
            raise
    
    def process_data_and_get_embeddings(self, file_name, has_labels=True):
        """Process data and extract embeddings"""
        df = self.load_data(file_name)
        
        all_embeddings = []
        all_model_predictions = []
        all_labels = []
        
        # Get indices for submission - use DataFrame index if 'index' column doesn't exist
        if 'index' in df.columns:
            all_indices = df['index'].tolist()
        else:
            # Use the DataFrame's own index as a fallback
            log.info("No 'index' column found, using DataFrame index instead")
            all_indices = df.index.tolist()
        
        # Process in batches
        for i in tqdm(range(0, len(df), self.batch_size), desc=f"Extracting embeddings from {file_name}"):
            batch_df = df.iloc[i:i+self.batch_size]
            texts = batch_df['text'].tolist()
            embeddings, model_predictions = self.get_embeddings(texts)
            all_embeddings.append(embeddings)
            all_model_predictions.append(model_predictions)
            
            # If the dataset has labels (train), collect them
            if has_labels:
                if isinstance(batch_df['rate'].iloc[0], str):
                    # Convert string labels to indices using cat2idx mapping
                    labels = [self.cat2idx.get(str(label), 0) for label in batch_df['rate']]
                else:
                    # Use numeric labels directly
                    labels = batch_df['rate'].tolist()
                all_labels.extend(labels)
        
        # Concatenate all batches
        embeddings = np.vstack(all_embeddings)
        model_predictions = np.concatenate(all_model_predictions)
        
        result = {
            'embeddings': embeddings,
            'model_predictions': model_predictions,
            'indices': all_indices
        }
        
        if has_labels:
            result['labels'] = all_labels
            
        return result
    
    def train_knn_model(self, train_embeddings, train_labels):
        """Train KNN model on embeddings"""
        log.info(f"Training KNN model with {self.n_neighbors} neighbors on {len(train_embeddings)} samples")
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.knn.fit(train_embeddings, train_labels)
        log.info("KNN model trained successfully")
    
    def predict_with_knn(self, embeddings):
        """Make predictions using the trained KNN model"""
        if self.knn is None:
            raise ValueError("KNN model not trained yet. Call train_knn_model first.")
        
        log.info(f"Making predictions with KNN on {len(embeddings)} samples")
        knn_predictions = self.knn.predict(embeddings)
        return knn_predictions
    
    def evaluate_predictions(self, true_labels, knn_predictions, model_predictions):
        """Compare KNN and model predictions against true labels"""
        log.info("Evaluating predictions...")
        
        # Calculate metrics for KNN
        knn_accuracy = accuracy_score(true_labels, knn_predictions)
        knn_precision = precision_score(true_labels, knn_predictions, average='weighted')
        knn_recall = recall_score(true_labels, knn_predictions, average='weighted')
        knn_f1 = f1_score(true_labels, knn_predictions, average='weighted')
        
        # Calculate metrics for original model
        model_accuracy = accuracy_score(true_labels, model_predictions)
        model_precision = precision_score(true_labels, model_predictions, average='weighted')
        model_recall = recall_score(true_labels, model_predictions, average='weighted')
        model_f1 = f1_score(true_labels, model_predictions, average='weighted')
        
        # Print results
        log.info(f"KNN Metrics: Accuracy={knn_accuracy:.4f}, Precision={knn_precision:.4f}, "
                 f"Recall={knn_recall:.4f}, F1={knn_f1:.4f}")
        log.info(f"Model Metrics: Accuracy={model_accuracy:.4f}, Precision={model_precision:.4f}, "
                 f"Recall={model_recall:.4f}, F1={model_f1:.4f}")
        
        # Calculate difference
        acc_diff = knn_accuracy - model_accuracy
        log.info(f"KNN vs Model accuracy difference: {acc_diff:.4f} ({'better' if acc_diff > 0 else 'worse'})")
        
        # Return metrics for plotting
        return {
            'knn': {
                'accuracy': knn_accuracy,
                'precision': knn_precision,
                'recall': knn_recall,
                'f1': knn_f1
            },
            'model': {
                'accuracy': model_accuracy,
                'precision': model_precision,
                'recall': model_recall,
                'f1': model_f1
            }
        }
    
    def plot_confusion_matrix(self, true_labels, knn_predictions, model_predictions):
        """Plot confusion matrices for KNN and model predictions"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # KNN confusion matrix
        cm_knn = confusion_matrix(true_labels, knn_predictions)
        sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('KNN Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        
        # Model confusion matrix
        cm_model = confusion_matrix(true_labels, model_predictions)
        sns.heatmap(cm_model, annot=True, fmt='d', cmap='Blues', ax=axes[1])
        axes[1].set_title('Model Confusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')
        
        # Save the figure
        plt.tight_layout()
        save_path = os.path.join(self.path_to_plots, f"confusion_matrices_{self.date.strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(save_path, dpi=300)
        log.info(f"Confusion matrices saved to {save_path}")
        plt.close()
    
    def plot_metrics_comparison(self, metrics):
        """Plot comparison of metrics between KNN and original model"""
        metrics_names = ['accuracy', 'precision', 'recall', 'f1']
        knn_values = [metrics['knn'][m] for m in metrics_names]
        model_values = [metrics['model'][m] for m in metrics_names]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, knn_values, width, label='KNN')
        rects2 = ax.bar(x + width/2, model_values, width, label='Original Model')
        
        ax.set_ylabel('Score')
        ax.set_title('Performance comparison: KNN vs Original Model')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.legend()
        
        # Add values on top of bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        
        # Save the figure
        plt.tight_layout()
        save_path = os.path.join(self.path_to_plots, f"metrics_comparison_{self.date.strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(save_path, dpi=300)
        log.info(f"Metrics comparison saved to {save_path}")
        plt.close()
    
    def reduce_dimensionality(self, embeddings):
        """Reduce dimensionality using PCA"""
        log.info(f"Reducing dimensionality from {embeddings.shape[1]} to 2")
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)
        explained_variance = pca.explained_variance_ratio_.sum()
        log.info(f"Explained variance: {explained_variance:.2f}")
        return reduced_embeddings

    def visualize_embeddings(self, reduced_embeddings, labels, title, filename):
        """Create a scatter plot of the embeddings colored by class"""
        plt.figure(figsize=(12, 10))
        
        # Create a scatter plot
        scatter = plt.scatter(
            reduced_embeddings[:, 0], 
            reduced_embeddings[:, 1], 
            c=labels, 
            cmap='tab10', 
            alpha=0.7,
            s=15
        )
        
        # Add a color bar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Class')
        
        # Set labels for actual class names if idx2cat is available
        if self.idx2cat:
            # Get unique classes in the current dataset
            unique_classes = np.unique(labels)
            
            # Set the ticks to match the unique classes
            cbar.set_ticks(unique_classes)
            
            # Create labels (with fallback for missing entries)
            cbar_labels = []
            for idx in unique_classes:
                if isinstance(idx, np.integer):
                    idx = int(idx)
                if idx in self.idx2cat:
                    cbar_labels.append(f"{idx}: {self.idx2cat[idx]}")
                else:
                    cbar_labels.append(f"{idx}: Unknown")
                    
            # Set tick labels
            cbar.set_ticklabels(cbar_labels)
        
        plt.title(title)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.tight_layout()
        
        # Save the figure
        save_path = os.path.join(self.path_to_plots, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log.info(f"Plot saved to {save_path}")
        plt.close()

    def create_submission(self, test_indices, knn_predictions):
        """Create a submission file with KNN predictions"""
        # Map numeric predictions back to category labels with fallback
        mapped_predictions = []
        for pred in knn_predictions:               
            # Use a fallback for predictions outside our mapping
            mapped_predictions.append(pred)
        
        # Create submission DataFrame
        submission_df = pd.DataFrame({
            'index': test_indices,
            'rate': mapped_predictions
        })
        
        # Save submission to file
        timestamp = self.date.strftime("%Y%m%d_%H%M%S")
        submission_path = os.path.join(self.path_to_submission, f"knn_submission_{timestamp}.csv")
        submission_df.to_csv(submission_path, index=False)
        
        log.info(f"KNN Submission saved to {submission_path}")
        return submission_path

    def run(self):
        """Run the full KNN prediction pipeline"""
        # Load model
        self.load_model()
        
        # Process train data
        log.info("Processing train data...")
        train_data = self.process_data_and_get_embeddings('train.csv', has_labels=True)
        
        # Process test data
        log.info("Processing test data...")
        test_data = self.process_data_and_get_embeddings('test.csv', has_labels=False)
        
        # Train KNN model on embeddings
        self.train_knn_model(train_data['embeddings'], train_data['labels'])
        
        # Make predictions on train data with KNN
        log.info("Making KNN predictions on train data...")
        train_knn_predictions = self.predict_with_knn(train_data['embeddings'])
        
        # Evaluate on train data
        metrics = self.evaluate_predictions(
            train_data['labels'], 
            train_knn_predictions, 
            train_data['model_predictions']
        )
        
        # Plot confusion matrices
        self.plot_confusion_matrix(
            train_data['labels'], 
            train_knn_predictions, 
            train_data['model_predictions']
        )
        
        # Plot metrics comparison
        self.plot_metrics_comparison(metrics)
        
        # Make predictions on test data with KNN
        log.info("Making KNN predictions on test data...")
        test_knn_predictions = self.predict_with_knn(test_data['embeddings'])
        
        # Visualize embeddings with dimensionality reduction
        log.info("Visualizing embeddings with KNN predictions...")
        
        # Combine embeddings for better PCA fit
        combined_embeddings = np.vstack([train_data['embeddings'], test_data['embeddings']])
        reduced_combined = self.reduce_dimensionality(combined_embeddings)
        
        # Split back to train and test
        reduced_train = reduced_combined[:len(train_data['embeddings'])]
        reduced_test = reduced_combined[len(train_data['embeddings']):]
        
        # Visualize train data with true labels
        self.visualize_embeddings(
            reduced_train, 
            train_data['labels'], 
            'Train Data Embeddings (True Labels)', 
            f'train_embeddings_true_labels_{self.date.strftime("%Y%m%d_%H%M%S")}.png'
        )
        
        # Visualize train data with KNN predictions
        self.visualize_embeddings(
            reduced_train, 
            train_knn_predictions, 
            'Train Data Embeddings (KNN Predictions)', 
            f'train_embeddings_knn_predictions_{self.date.strftime("%Y%m%d_%H%M%S")}.png'
        )
        
        # Visualize train data with model predictions
        self.visualize_embeddings(
            reduced_train, 
            train_data['model_predictions'], 
            'Train Data Embeddings (Model Predictions)', 
            f'train_embeddings_model_predictions_{self.date.strftime("%Y%m%d_%H%M%S")}.png'
        )
        
        # Visualize test data with KNN predictions
        self.visualize_embeddings(
            reduced_test, 
            test_knn_predictions, 
            'Test Data Embeddings (KNN Predictions)', 
            f'test_embeddings_knn_predictions_{self.date.strftime("%Y%m%d_%H%M%S")}.png'
        )
        
        # Create submission file with KNN predictions
        self.create_submission(test_data['indices'], test_knn_predictions)
        
        log.info("KNN prediction pipeline completed")


if __name__ == "__main__":
    from src import path_to_config
    from src.utils.config_parser import ConfigParser
    from env import Env

    env = Env()
    config = ConfigParser.parse(path_to_config())
    
    # Get KNN prediction config
    knn_config = config.get('KNNEmbed2Cat', {})
    
    # Create KNN prediction object
    knn = KNNEmbed2Cat(
        path_to_data=env.__getattr__("DATA_PATH"),
        path_to_weights=env.__getattr__("WEIGHTS_PATH"),
        path_to_plots=env.__getattr__("PLOTS_PATH") if hasattr(env, "PLOTS_PATH") else "./plots",
        path_to_submission=env.__getattr__("SUBMISSION_PATH") if hasattr(env, "SUBMISSION_PATH") else "./submissions",
        **knn_config
    )
    
    # Run the KNN prediction pipeline
    knn.run()
