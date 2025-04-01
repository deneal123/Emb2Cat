import os
import torch
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from transformers import AutoModel, AutoTokenizer
from datetime import datetime
from src import project_path
from src.modelling.classification_head import TextClassificationModel
from src.utils.custom_logging import setup_logging

log = setup_logging()


@dataclass
class InferEmbed2Cat:
    path_to_data: str = "./data"
    path_to_weights: str = "./weights"
    path_to_submission: str = "./submissions"
    name_model: str = "cat"
    model_path_id: str = "tabularisai/multilingual-sentiment-analysis"
    use_device: str = "cuda"
    num_classes: int = 5
    batch_size: int = 16
    convert_to_numeric: bool = False  # New parameter to control numeric vs string output

    def __post_init__(self):
        self.date = datetime.now()
        self.path_to_data = Path(os.path.join(project_path, self.path_to_data))
        self.path_to_weights = Path(os.path.join(project_path, self.path_to_weights, 'embed2cat'))
        self.path_to_submission = Path(os.path.join(project_path, self.path_to_submission))
        
        self.model = None
        self.tokenizer = None
        self.idx2cat = None  # Mapping from model indices to category names
        
        # Set device
        if not self.use_device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.use_device == "cpu":
            self.device = torch.device("cpu")
        elif self.use_device == "cuda":
            self.device = torch.device("cuda")
            
        # Create submission directory if it doesn't exist
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
    
    def load_test_data(self):
        """Load test data from CSV file"""
        test_path = os.path.join(self.path_to_data, "test.csv")
        log.info(f"Loading test data from {test_path}")
        
        try:
            test_df = pd.read_csv(test_path)
            log.info(f"Loaded {len(test_df)} test samples")
            return test_df
        except Exception as e:
            log.error(f"Error loading test data: {e}")
            raise
    
    def predict(self, texts):
        """Make predictions for a batch of texts"""
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
            
            # Get predictions
            logits = self.model(inputs)
            _, predictions = torch.max(logits, 1)
            
            return predictions.cpu().numpy()
    
    def map_predictions_to_categories(self, predictions):
        """Convert model prediction indices to actual category names"""
        mapped_predictions = []
        
        for pred in predictions:
            category_name = self.idx2cat[pred]
            
            # Convert to numeric if requested
            if self.convert_to_numeric:
                try:
                    mapped_predictions.append(int(category_name))
                except ValueError:
                    mapped_predictions.append(category_name)
            else:
                mapped_predictions.append(category_name)
                
        return mapped_predictions
    
    def create_submission(self):
        """Create submission file from test predictions"""
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Load test data
        test_df = self.load_test_data()
        
        all_predictions = []
        
        # Process in batches
        for i in tqdm(range(0, len(test_df), self.batch_size), desc="Generating predictions"):
            batch_df = test_df.iloc[i:i+self.batch_size]
            texts = batch_df['text'].tolist()
            predictions = self.predict(texts)
            all_predictions.extend(predictions)
        
        # Map predictions to actual category names
        mapped_predictions = self.map_predictions_to_categories(all_predictions)
        
        # Create submission DataFrame
        submission_df = pd.DataFrame({
            'index': test_df['index'],
            'rate': mapped_predictions
        })
        
        # Save submission to file
        timestamp = self.date.strftime("%Y%m%d_%H%M%S")
        submission_path = os.path.join(self.path_to_submission, f"submission_{timestamp}.csv")
        submission_df.to_csv(submission_path, index=False)
        
        log.info(f"Submission saved to {submission_path}")
        return submission_path


if __name__ == "__main__":
    from src import path_to_config
    from src.utils.config_parser import ConfigParser
    from env import Env

    env = Env()
    config = ConfigParser.parse(path_to_config())
    
    # Get inference config (can reuse training config or create a separate one)
    infer_config = config.get('InferParamEmbed2Cat', {})
    
    # Create inference object
    infer = InferEmbed2Cat(
        path_to_data=env.__getattr__("DATA_PATH"),
        path_to_weights=env.__getattr__("WEIGHTS_PATH"),
        path_to_submission=env.__getattr__("SUBMISSION_PATH") if hasattr(env, "SUBMISSION_PATH") else "./submissions",
        **infer_config
    )
    
    # Create submission file
    submission_path = infer.create_submission()
    print(f"Submission created at: {submission_path}")
