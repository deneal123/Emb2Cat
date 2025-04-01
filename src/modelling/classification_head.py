import torch
import torch.nn as nn


# Определяем полную модель с классификационной головой
class TextClassificationModel(nn.Module):
    def __init__(self, base_model, num_classes, unfreeze_layers=0):
        super(TextClassificationModel, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.15)
        self.ks1 = nn.Linear(base_model.config.hidden_size, 512)
        self.ks2 = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, num_classes)
        
        # Сначала замораживаем все параметры базовой модели
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Если указано число слоев для разморозки, размораживаем последние N слоев
        if unfreeze_layers > 0:
            # Определяем архитектуру модели и получаем доступ к слоям
            if hasattr(self.base_model, 'encoder') and hasattr(self.base_model.encoder, 'layer'):
                # BERT, RoBERTa и другие стандартные модели
                layers = self.base_model.encoder.layer
                print(f"Detected standard transformer model with {len(layers)} layers")
            elif hasattr(self.base_model, 'transformer') and hasattr(self.base_model.transformer, 'layer'):
                # DistilBERT и похожие модели
                layers = self.base_model.transformer.layer
                print(f"Detected DistilBERT-like model with {len(layers)} layers")
            else:
                print("Model architecture not recognized for layer unfreezing")
                layers = []
            
            # Размораживаем последние unfreeze_layers слоев, если они есть
            if len(layers) > 0:
                # Проверяем, что unfreeze_layers не больше количества слоев
                actual_unfreeze = min(unfreeze_layers, len(layers))
                for layer in layers[-actual_unfreeze:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                print(f"Unfrozen last {actual_unfreeze} transformer layers")
            
            # Размораживаем эмбеддинги (опционально)
            if hasattr(self.base_model, 'embeddings'):
                # Размораживаем только если запросили разморозить большую часть модели
                if unfreeze_layers > len(layers) // 2:
                    for param in self.base_model.embeddings.parameters():
                        param.requires_grad = True
                    print("Unfrozen embeddings layer")
            
            # Некоторые модели используют word_embedding вместо embeddings
            elif hasattr(self.base_model, 'word_embedding'):
                if unfreeze_layers > len(layers) // 2:
                    for param in self.base_model.word_embedding.parameters():
                        param.requires_grad = True
                    print("Unfrozen word_embedding layer")
        
    def forward(self, inputs):
        # Получаем выходные данные из базовой модели
        outputs = self.base_model(**inputs)
        # Берем [CLS] токен или среднее по всем токенам
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        # Получаем логиты через классификационную голову
        opt1 = self.ks1(pooled_output)
        opt1 = torch.relu(opt1)
        opt1 = self.dropout(opt1)
        opt2 = self.ks2(opt1)
        opt2 = torch.relu(opt2)
        opt2 = self.dropout(opt2)
        logits = self.classifier(opt2)
        return logits
