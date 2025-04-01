import torch
import torch.nn as nn


class TextClassificationModel(nn.Module):
    def __init__(self, base_model, num_classes, hidden_sizes=[512, 256]):
        super(TextClassificationModel, self).__init__()
        self.base_model = base_model
        
        input_size = base_model.config.hidden_size
        
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        
        self.classifier = nn.Linear(hidden_sizes[1], num_classes)
        
    def forward(self, inputs):
        outputs = self.base_model(**inputs)
        pooled_output = outputs.last_hidden_state[:, 0]
        
        x = self.fc1(pooled_output)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        logits = self.classifier(x)
        return logits
