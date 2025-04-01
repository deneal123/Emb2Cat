import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Callable
from pathlib import Path
import json

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

from src import path_to_project
from src.utils.custom_logging import setup_logging
from src.utils.seed import seed_everything
from src.utils.config_parser import ConfigParser
from src import path_to_config


log = setup_logging()
config = ConfigParser.parse(path_to_config())


# Обновляем функцию collate_fn, добавляя возможность аугментации
def collate_fn(batch):

    process_batch = {
        "texts": [],
        "categories": [],
        "category_ids": []
    }

    for i, sample in enumerate(batch):
        text = sample['text']
        category = sample['category']
        category_id = sample['category_id']

        # Здесь можно добавить аугментацию
        # text = 

        # ...

        # Добавляем данные в batch
        process_batch['texts'].append(text)
        process_batch['categories'].append(category)
        process_batch['category_ids'].append(category_id)

    # Преобразуем category_id в один тензор после итерации
    process_batch['category_ids'] = torch.tensor(process_batch['category_ids'])
    # Не пытаемся стекать тексты, просто оставляем список
    # process_batch['texts'] = torch.stack(process_batch['texts'])

    return process_batch


def get_datasets(data_folder: str,
                 val_size: float = 0,
                 test_size: float = 0,
                 separator: str = 'SEP',
                 seed: int = 17
                 ):
    """
    Создает тренировочный, валидационный и тестовый датасеты на основе указанного CSV файла

    Args:
        data_folder (str): Путь к директории с данными
        val_size (float): Доля данных для валидации, в диапазоне [0, 1)
        test_size (float): Доля данных для тестирования, в диапазоне [0, 1)
        separator (str): Разделитель для тегов, если несколько тегов
        categories (List[str]): Список категорий для фильтрации данных
        subcategories (List[str]): Список подкатегорий для фильтрации данных

    Returns:
        Tuple: Датасеты для обучения, валидации (если есть) и тестирования
    """
    seed_everything(seed)

    assert 0 <= val_size < 1, "'val_size' should be in the range [0, 1)"
    assert 0 <= test_size < 1, "'test_size' should be in the range [0, 1)"

    if test_size == 0 and val_size > 0:
        test_size = val_size
        val_size = 0

    path = os.path.join(data_folder, 'train.csv')
    metadata = pd.read_csv(path)

    # Разделение на тренировочный и тестовый датасет
    train_metadata, test_metadata = train_test_split(
        metadata,
        test_size=test_size,
        stratify=metadata['rate'].values,
        random_state=seed
    )

    # Если валидационный датасет указан, делаем разделение
    if val_size > 0:
        train_metadata, val_metadata = train_test_split(
            train_metadata,
            test_size=val_size / (1 - test_size),
            stratify=train_metadata['rate'].values,
            random_state=seed
        )
        val_dataset = VideoDataset(data_folder=data_folder,
                                   metadata=val_metadata,
                                   separator=separator,
                                   set_name="val")
    else:
        val_dataset = None

    # Создание датасетов
    train_dataset = VideoDataset(data_folder=data_folder, metadata=train_metadata, separator=separator)
    test_dataset = VideoDataset(data_folder=data_folder, metadata=test_metadata, separator=separator, set_name="test")

    if val_dataset is not None:
        return train_dataset, val_dataset, test_dataset
    else:
        return train_dataset, test_dataset


class VideoDataset(Dataset):

    def __init__(
            self,
            data_folder: str,
            metadata: pd.DataFrame,
            set_name: str = 'train',
            separator: str = 'SEP',
    ):

        self.metadata = metadata
        self.separator = separator
        self.set_name = set_name
        self.data_folder = data_folder

        self._build_target()

    def _build_target(self) -> None:
        """
            Builds label encodings for categories and tags
        """
        self.categories = self.metadata.rate.values

        # LabelEncoding for unique categories
        unique_categories = np.unique(self.categories)  # Уникальные категории
        self.cat2idx = {category: idx for idx, category in enumerate(unique_categories)}
        self.idx2cat = {idx: category for idx, category in enumerate(unique_categories)}
        
        with open(os.path.join(self.data_folder, 'cat2idx.json'), 'w') as f:
            json.dump({idx: str(category) for idx, category in enumerate(unique_categories)}, f)

        self.num_categories = len(unique_categories)

        log.info(f'''{self.set_name.upper()} INFO\nTotal: {self.num_categories} categories''')

    def __len__(self) -> int:
        return len(self.metadata)

    @staticmethod
    def truncate_string(text: str, max_length: int) -> str:
        try:
            if len(text) > max_length:
                return text[:max_length]
            return text
        except Exception as e:
            return ''

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
        category = self.categories[idx]
        category_id = self.cat2idx[category]
        text = self.truncate_string(self.metadata['text'].values[idx], 2048)

        return {
            "text": text,
            "category": category,
            "category_id": category_id
        }
