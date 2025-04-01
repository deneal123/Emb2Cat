import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
class ClusterEmbed2Cat:
    path_to_data: str = "./data"
    path_to_weights: str = "./weights"
    path_to_plots: str = "./plots/embeddings"
    name_model: str = "cat"
    model_path_id: str = "tabularisai/multilingual-sentiment-analysis"
    use_device: str = "cuda"
    num_classes: int = 5
    batch_size: int = 16
    n_components: int = 2  # Number of components for PCA

    def __post_init__(self):
        self.date = datetime.now()
        self.path_to_data = Path(os.path.join(project_path, self.path_to_data))
        self.path_to_weights = Path(os.path.join(project_path, self.path_to_weights, 'embed2cat'))
        self.path_to_plots = Path(os.path.join(project_path, self.path_to_plots))
        
        self.model = None
        self.tokenizer = None
        self.idx2cat = None
        
        # Set device
        if not self.use_device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.use_device == "cpu":
            self.device = torch.device("cpu")
        elif self.use_device == "cuda":
            self.device = torch.device("cuda")
            
        # Create plots directory if it doesn't exist
        os.makedirs(self.path_to_plots, exist_ok=True)
        
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
        """Load the trained model and tokenizer"""
        log.info(f"Loading model from {self.path_to_weights}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path_id)
        
        # Create model architecture
        base_model = AutoModel.from_pretrained(self.model_path_id)    
        self.model = TextClassificationModel(base_model, self.num_classes)
        
        # Load model weights
        model_path = os.path.join(self.path_to_weights, f"{self.name_model}.pt")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            log.info("Model loaded successfully")
        else:
            log.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
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
            
            # Get predictions for coloring
            # Forward through the full model
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
        all_predictions = []
        all_labels = []
        
        # Process in batches
        for i in tqdm(range(0, len(df), self.batch_size), desc=f"Extracting embeddings from {file_name}"):
            batch_df = df.iloc[i:i+self.batch_size]
            texts = batch_df['text'].tolist()
            embeddings, predictions = self.get_embeddings(texts)
            all_embeddings.append(embeddings)
            all_predictions.append(predictions)
            
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
        predictions = np.concatenate(all_predictions)
        
        return embeddings, predictions, all_labels if has_labels else None
    
    def reduce_dimensionality(self, embeddings):
        """Reduce dimensionality using PCA"""
        log.info(f"Reducing dimensionality from {embeddings.shape[1]} to {self.n_components}")
        pca = PCA(n_components=self.n_components)
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
    
    def run(self):
        """Run the full pipeline: extract embeddings, reduce dimensionality, and visualize"""
        # Load model
        self.load_model()
        
        # Process train data
        log.info("Processing train data...")
        train_embeddings, train_predictions, train_labels = self.process_data_and_get_embeddings('train.csv', has_labels=True)
        
        # Process test data
        log.info("Processing test data...")
        test_embeddings, test_predictions, _ = self.process_data_and_get_embeddings('test.csv', has_labels=False)
        
        # Combine embeddings for dimensionality reduction
        combined_embeddings = np.vstack([train_embeddings, test_embeddings])
        reduced_combined = self.reduce_dimensionality(combined_embeddings)
        
        # Split back to train and test
        reduced_train = reduced_combined[:len(train_embeddings)]
        reduced_test = reduced_combined[len(train_embeddings):]
        
        # Visualize train data with true labels
        self.visualize_embeddings(
            reduced_train, 
            train_labels, 
            'Train Data Embeddings (True Labels)', 
            f'train_embeddings_true_labels_{self.date.strftime("%Y%m%d_%H%M%S")}.png'
        )
        
        # Visualize train data with predicted labels
        self.visualize_embeddings(
            reduced_train, 
            train_predictions, 
            'Train Data Embeddings (Predicted Labels)', 
            f'train_embeddings_predicted_labels_{self.date.strftime("%Y%m%d_%H%M%S")}.png'
        )
        
        # Visualize test data with predicted labels
        self.visualize_embeddings(
            reduced_test, 
            test_predictions, 
            'Test Data Embeddings (Predicted Labels)', 
            f'test_embeddings_predicted_labels_{self.date.strftime("%Y%m%d_%H%M%S")}.png'
        )
        
        log.info("Embedding visualization completed")


if __name__ == "__main__":
    from src import path_to_config
    from src.utils.config_parser import ConfigParser
    from env import Env

    env = Env()
    config = ConfigParser.parse(path_to_config())
    
    # Get embedding visualization config
    cluster_config = config.get('ClusterEmbed2Cat', {})
    
    # Create cluster object
    cluster = ClusterEmbed2Cat(
        path_to_data=env.__getattr__("DATA_PATH"),
        path_to_weights=env.__getattr__("WEIGHTS_PATH"),
        path_to_plots=env.__getattr__("PLOTS_PATH") if hasattr(env, "PLOTS_PATH") else "./plots",
        **cluster_config
    )
    
    # Run the embedding visualization
    cluster.run()
