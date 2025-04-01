import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src import project_path
from src.utils.custom_logging import setup_logging
from pathlib import Path
from src.modelling.dataset import get_datasets, collate_fn
from datetime import datetime
from functools import partial
import torch.nn.functional as F
from src.utils.save_param import save_model, save_metrics_train, save_metrics_test
from src.utils.create_dir import create_directories_if_not_exist
from dataclasses import dataclass
from transformers import AutoModel, AutoTokenizer
from src.modelling.classification_head import TextClassificationModel
from src.modelling.focal_loss import FocalLoss

log = setup_logging()


@dataclass
class GraduateEmbed2CatSubcat:
    path_to_data: str = "./data"
    path_to_weights: str = "./weights"
    path_to_metrics: str = "./metrics"
    name_model: str = "cat",
    model_path_id: str = "tabularisai/multilingual-sentiment-analysis",
    use_device: str = "cuda"
    start_learning_rate: float = 0.0001
    list_no_include_cat: list = None
    num_classes: int = 5
    batch_size: int = 16
    num_workers: int = 0
    pin_memory: bool = True
    num_epochs: int = 10
    name_optimizer: str = "Adam"
    val_size: float = 0.1
    test_size: float = 0.1
    seed: int = 17
    unfreeze_layers: int = 3  # Новый параметр для размораживания слоев

    def __post_init__(self):
        self.date = datetime.now()
        self.name_model = self.name_model if self.name_model else None
        self.path_to_data = Path(os.path.join(project_path, self.path_to_data))
        self.path_to_weights = Path(os.path.join(project_path, self.path_to_weights), 'embed2cat')
        self.path_to_metrics_train = Path(os.path.join(project_path, self.path_to_metrics, 'embed2cat'))
        self.path_to_metrics_test = Path(os.path.join(project_path, self.path_to_metrics, 'embed2cat'))

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.cat_total_samples = None
        self.cat_targets = None
        self.class_counts = None
        self.class_weights = None
        self.cat_criterion = None
        self.scheduler = None
        self.transform = None
        self.optimizer = None
        self.model = None
        self.checkpoint = None

        # Перемещение модели на GPU, если CUDA доступен
        if not self.use_device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.use_device == "cpu":
            self.device = torch.device("cpu")
        elif self.use_device == "cuda":
            self.device = torch.device("cuda")

        if self.device == "cpu":
            self.pin_memory = False

        # Создаем директории для сохранения весов и метрик
        create_directories_if_not_exist([self.path_to_weights,
                                         self.path_to_metrics_train,
                                         self.path_to_metrics_test])

    def graduate(self):
        # Получаем генераторы обучения, валидации и теста
        self.get_loaders()
        # Загружаем модель
        self.get_model()
        # Получаем веса классов
        self.get_classes_weights()
        # Определяем оптимизатор, функцию потерь и планировщик
        self.get_opt_crit_sh()
        # Загружаем чекпоинт
        self.load_checkpoint()
        # Выводим информацию
        print(self.__str__())
        # Обучаем
        self.train_model()
        # Тестируем
        self.evaluate_model()

    def __str__(self):
        log.info(f"Определенное устройство: {self.use_device}")
        log.info(f"Количество эпох обучения {self.num_epochs}")
        log.info(f"Размер пакета: {self.batch_size}")
        log.info(f"Выбранная модель: {self.model_path_id}")
        log.info(f"Данные загружены из директории: {self.path_to_data}")
        log.info(f"Количество каждой категории: {self.num_classes}")
        log.info(f"Веса каждого класса: {self.class_weights}")
        log.info(f"Выбранный оптимизатор: {self.name_optimizer}")
        return """"""

    # Функция для загрузки данных
    def get_loaders(self):
        # Определяем класс video_dataset
        self.train_dataset, self.valid_dataset, self.test_dataset = get_datasets(self.path_to_data,
                                                                                 val_size=self.val_size,
                                                                                 test_size=self.test_size,
                                                                                 seed=self.seed)
        # Инициализируем DataLoader
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       collate_fn=partial(collate_fn),
                                       num_workers=self.num_workers,
                                       pin_memory=self.pin_memory)
        self.valid_loader = DataLoader(self.valid_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=False,
                                       collate_fn=partial(collate_fn),
                                       num_workers=self.num_workers,
                                       pin_memory=self.pin_memory)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      collate_fn=partial(collate_fn),
                                      num_workers=self.num_workers,
                                      pin_memory=self.pin_memory)

        self.classes = np.unique(self.train_dataset.categories)

    def get_model(self):
        # Создаем токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path_id)
        # Создаем базовую модель
        base_model = AutoModel.from_pretrained(self.model_path_id)    
        self.model = TextClassificationModel(base_model, self.num_classes, unfreeze_layers=self.unfreeze_layers)
        self.model.to(self.device)

    def get_classes_weights(self):
        # Получение категорий из тренировочного датасета
        self.cat_targets = self.train_dataset.categories
        # Преобразование категорий в числовые индексы (если они не числа)
        if isinstance(self.cat_targets[0], str):
            self.cat_targets = np.array([self.train_dataset.cat2idx[cat] for cat in self.cat_targets])
        self.cat_total_samples = len(self.cat_targets)
        # Получение количества примеров для каждого класса
        self.class_counts = np.bincount(self.cat_targets)[1:]
        # Вычисление весов классов
        self.class_weights = torch.tensor([self.cat_total_samples / count for count in self.class_counts],
                                          dtype=torch.float)
        self.class_weights = self.class_weights.to(self.device)

    def get_opt_crit_sh(self):
        # Определение функции потерь с учетом весов классов
        self.class_weights = self.class_weights if self.class_weights is not None else None
        # self.cat_criterion = nn.BCEWithLogitsLoss(weight=self.class_weights)
        self.cat_criterion = FocalLoss(weight=self.class_weights, gamma=2)
        self.optimizer = optim.__dict__[f"{self.name_optimizer}"](self.model.parameters(), lr=self.start_learning_rate)
        # Создание планировщика LR
        # ReduceLROnPlateau уменьшает скорость обучения, когда метрика перестает уменьшаться
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=2, verbose=True)

    def load_checkpoint(self):
        path = os.path.join(self.path_to_weights, f"{self.name_model}.pt")
        try:
            if os.path.isfile(path):
                self.checkpoint = torch.load(path, map_location=self.device, weights_only=True)
                try:
                    self.model.load_state_dict(self.checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
                    log.info("Веса успешно загружены")
                except Exception as ex:
                    log.info("Ошибка загрузки предварительно обученной модели", exc_info=ex)
            else:
                log.info("Не найден файл с моделью")
        except Exception as ex:
            log.info("Ошибка загрузки предварительно обученной модели", exc_info=ex)

    # Функция для обучения модели с валидацией
    def train_model(self):
        train_loss_values = []
        valid_loss_values = []
        f1_values = []

        for epoch in range(self.num_epochs):

            # Вычисление loss на тренировочном датасете
            self.model.train()
            train_loss = 0.0

            with tqdm(total=len(self.train_loader)) as pbar_train:
                for index, batch in enumerate(self.train_loader):
                    # Распаковка данных
                    texts = batch["texts"]
                    categories = batch["categories"]
                    category_ids = batch["category_ids"].to(self.device)
                
                    # Токенизируем тексты
                    encoded_inputs = self.tokenizer([str(text) for text in texts], 
                                                   padding=True, 
                                                   truncation=True, 
                                                   return_tensors="pt")
                    # Перемещаем на нужное устройство
                    inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}

                    # Определяем метки категорий
                    cat_labels = category_ids
                    # Создаем one-hot тензоры на правильном устройстве
                    cat_labels_one_hot = torch.zeros((cat_labels.size(0), self.num_classes), device=self.device)
                    # Заполняем one-hot тензоры
                    cat_labels_one_hot.scatter_(1, cat_labels.unsqueeze(1), 1)

                    # Обучаем модель
                    category_logits = self.model(inputs)
                    loss = self.cat_criterion(category_logits, cat_labels_one_hot)

                    self.optimizer.zero_grad()

                    train_loss += loss.item() * self.batch_size
                    loss.backward()
                    self.optimizer.step()

                    # Обновляем бар
                    pbar_train.set_description(f"(Train)")
                    pbar_train.unit = " sample"
                    pbar_train.set_postfix(epoch=(epoch + 1), loss=train_loss / ((index + 1) * self.batch_size))
                    pbar_train.update(1)

            # Вычисление loss на валидационном датасете и метрик
            self.model.eval()
            valid_loss = 0.0
            best_f1 = 0.0
            all_cat_predictions = []
            all_cat_labels = []

            with torch.no_grad():
                with tqdm(total=len(self.valid_loader)) as pbar_valid:
                    for index, batch in enumerate(self.valid_loader):
                        # Распаковка данных
                        texts = batch["texts"]
                        categories = batch["categories"]
                        category_ids = batch["category_ids"].to(self.device)
                        
                        # Токенизируем тексты
                        encoded_inputs = self.tokenizer([str(text) for text in texts], 
                                                       padding=True, 
                                                       truncation=True, 
                                                       return_tensors="pt")
                        # Перемещаем на нужное устройство
                        inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}

                        # Определяем метки категорий
                        cat_labels = category_ids
                        # Создаем one-hot тензоры на правильном устройстве
                        cat_labels_one_hot = torch.zeros((cat_labels.size(0), self.num_classes), device=self.device)
                        # Заполняем one-hot тензоры
                        cat_labels_one_hot.scatter_(1, cat_labels.unsqueeze(1), 1)

                        # Валидируем модель
                        category_logits = self.model(inputs)
                        loss = self.cat_criterion(category_logits, cat_labels_one_hot)

                        valid_loss += loss.item() * self.batch_size

                        _, cat_predicted = torch.max(category_logits, 1)
                        all_cat_predictions.extend(cat_predicted.cpu().numpy())
                        all_cat_labels.extend(cat_labels.cpu().numpy())

                        # Обновляем бар
                        pbar_valid.set_description(f"(Valid)")
                        pbar_valid.unit = " sample"
                        pbar_valid.set_postfix(epoch=(epoch + 1), loss=valid_loss / ((index + 1) * self.batch_size),
                                               cat_lab=cat_labels.cpu().numpy()[0],
                                               cat_pre=cat_predicted.cpu().numpy()[0])
                        pbar_valid.update(1)

            epoch_train_loss = train_loss / len(self.train_dataset.categories)
            epoch_valid_loss = valid_loss / len(self.valid_dataset.categories)

            cat_accuracy_mean = accuracy_score(all_cat_labels, all_cat_predictions)
            accuracy = cat_accuracy_mean
            cat_precision = precision_score(all_cat_labels, all_cat_predictions, average='weighted', zero_division=0)
            precision = cat_precision
            cat_recall = recall_score(all_cat_labels, all_cat_predictions, average='weighted', zero_division=0)
            recall = cat_recall
            cat_f1 = f1_score(all_cat_labels, all_cat_predictions, average='weighted', zero_division=0)
            f1 = cat_f1
            log.info(f"Valid Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")

            # Мы хотим сохранить модель с лучшим значением f1-score
            if f1 > best_f1:
                save_model(self.path_to_weights,
                           self.name_model,
                           self.model.state_dict(),
                           self.optimizer.state_dict(),
                           self.num_epochs)

            f1_values.append(f1)

            # Сообщаем планировщику LR о текущей ошибке на валидационном наборе
            self.scheduler.step(epoch_valid_loss)

            # Добавление значений метрик в списки
            train_loss_values.append(epoch_train_loss)
            valid_loss_values.append(epoch_valid_loss)

            # Сохранение метрик
            save_metrics_train(
                self.path_to_metrics_train,
                train_loss_values,
                valid_loss_values,
                f1_values,
                "f1",
                self.date,
                self.name_model
            )

            log.info(
                f"\nEpoch {epoch + 1}/{self.num_epochs}, Training Loss: {epoch_train_loss}, Validation Loss: {epoch_valid_loss}")

        log.info("Тренировка завершена!")

    # Функция для оценки модели на тестовом датасете
    def evaluate_model(self):
        self.model.eval()
        correct = 0
        total = 0
        all_cat_predictions = []
        all_cat_labels = []

        # Initialize variables to track correct predictions for each class
        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes

        with torch.no_grad():
            with tqdm(total=len(self.test_loader)) as pbar_test:
                for index, batch in enumerate(self.test_loader):
                    # Распаковка данных
                    texts = batch["texts"]
                    categories = batch["categories"]
                    category_ids = batch["category_ids"].to(self.device)
                    
                    # Токенизируем тексты
                    encoded_inputs = self.tokenizer([str(text) for text in texts], 
                                                   padding=True, 
                                                   truncation=True, 
                                                   return_tensors="pt")
                    # Перемещаем на нужное устройство
                    inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}

                    # Определяем метки категорий
                    cat_labels = category_ids

                    # Тестируем модель
                    category_logits = self.model(inputs)

                    _, cat_predicted = torch.max(category_logits, 1)
                    all_cat_predictions.extend(cat_predicted.cpu().numpy())
                    all_cat_labels.extend(cat_labels.cpu().numpy())

                    # Calculate class-wise correct predictions
                    for i in range(len(cat_labels)):
                        cat_label = cat_labels[i].item()
                        class_correct[cat_label] += (cat_predicted[i] == cat_labels[i]).item()
                        correct += (cat_predicted[i] == cat_labels[i]).item()
                        class_total[cat_label] += 1
                        total += 1

                    # Обновляем бар
                    pbar_test.set_description(f"(Test)")
                    pbar_test.unit = " sample"
                    pbar_test.set_postfix(correct=correct, total=total)
                    pbar_test.update(1)

        cat_accuracy_mean = sum(class_correct) / sum(class_total)
        accuracy = cat_accuracy_mean
        cat_precision = precision_score(all_cat_labels, all_cat_predictions, average='weighted', zero_division=0)
        precision = cat_precision
        cat_recall = recall_score(all_cat_labels, all_cat_predictions, average='weighted', zero_division=0)
        recall = cat_recall
        cat_f1 = f1_score(all_cat_labels, all_cat_predictions, average='weighted', zero_division=0)
        f1 = cat_f1
        log.info(f"Test Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")

        class_acc_dir = {}
        sub_acc_dir = {}

        log.info(60 * "-")
        log.info("Accuracy for each category:")
        # Print accuracy for each class
        for i in range(len(self.classes)):
            class_acc = 100 * class_correct[i] / class_total[i] if class_total[i] != 0 else 0
            class_acc_dir[str(self.classes[i])] = class_acc
            log.info('Accuracy of %5s : %2d %%' % (self.classes[i], class_acc))
        log.info(60 * "-")

        acc_dir = {'category': class_acc_dir}

        # log.info(acc_dir)

        save_metrics_test(self.path_to_metrics_test,
                          self.name_model,
                          f1,
                          'f1',
                          self.date,
                          acc_dir)


if __name__ == "__main__":
    from src import path_to_config
    from src.utils.config_parser import ConfigParser
    from env import Env

    env = Env()
    config = ConfigParser.parse(path_to_config())

    train_config = config.get('TrainParamEmbed2Cat', {})

    graduate = GraduateEmbed2CatSubcat(path_to_data=env.__getattr__("DATA_PATH"),
                                       path_to_weights=env.__getattr__("WEIGHTS_PATH"),
                                       path_to_metrics=env.__getattr__("METRICS_PATH"),
                                       **train_config)
    graduate.graduate()

