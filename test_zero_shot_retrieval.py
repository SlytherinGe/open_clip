from PIL import Image
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from os.path import join, exists
import pickle
import time
from copy import deepcopy
import random

import torch
from src.open_clip.factory import create_model_and_transforms, get_tokenizer
from torch.utils.data import Dataset, DataLoader
from src.training.precision import get_autocast

def get_retrieval_dataloaders(args, preprocess_fn):
    
    if args.test_batch_size == None:
        args.test_batch_size = args.batch_size
        
    retrieval_dataloader = {}
    
class CsvDataset_customized(Dataset):
    def __init__(self, df, transforms, img_key, caption_key, tokenizer=None, return_img_path=False, 
                 root_data_dir=None):
        if root_data_dir is not None:
            df[img_key] = df[img_key].apply(lambda x: join(root_data_dir, x))
        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        self.tokenize = tokenizer
        self.return_img_path = return_img_path

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        if self.return_img_path:
            return images, texts, str(self.images[idx])
        return images, texts
    
class CsvDataset_image(Dataset):
    def __init__(self, df, transforms, img_key, return_img_path=False, root_data_dir=None):
        if root_data_dir is not None:
            df[img_key] = df[img_key].apply(lambda x: join(root_data_dir, x))
        self.images = df[img_key].tolist()
        self.transforms = transforms
        self.return_img_path = return_img_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        if self.return_img_path:
            return images, str(self.images[idx])
        return images
    
class CsvDataset_text(Dataset):
    def __init__(self, df, caption_key, tokenizer=None, return_original_text=False, root_data_dir=None, img_key=None):
        if root_data_dir is not None:
            df[img_key] = df[img_key].apply(lambda x: join(root_data_dir, x))
        self.captions = df[caption_key].tolist()
        self.tokenize = tokenizer
        self.return_original_text = return_original_text

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        original_text = str(self.captions[idx])
        texts = self.tokenize([original_text])[0]
        if self.return_original_text:
            return texts, original_text
        return texts

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def get_sample_identifier(filepath):
    return '/'.join(filepath.split('/')[-2:])


def test_zero_shot_retrieval(model, dataloader):
    return