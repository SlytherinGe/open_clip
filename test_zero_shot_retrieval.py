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
from test_zero_shot_classification import *
from benchmark_dataset_info import *
from open_clip.factory import get_tokenizer

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

def get_sample_identifier(filepath):
    return '/'.join(filepath.split('/')[-2:])


def get_caption_key(dataset_name):
    caption_key = 'title'
    if dataset_name in ['SkyScript']:
        caption_key = 'title_multi_objects'

    return caption_key

def get_retrieval_dataloader(args, preprocess_fn):
    
    if args.test_batch_size == None:
        args.test_batch_size = args.batch_size
        
    retrieval_dataloader = {}
    tokenizer = get_tokenizer(args.model)
    if args.datasets_for_retrieval is not None:
        for dataset_name in args.datasets_for_retrieval:
            data_csv_path = DATASET_CSV_FILE_PATH[dataset_name]
            caption_key = get_caption_key(dataset_name)
            df = pd.read_csv(data_csv_path)
            assert RETRIEVAL_DATASET_ROOT_DIR is not None, "Please specify the root directory of the retrieval datasets"
            df['filepath'] = df['filepath'].apply(lambda x: join(RETRIEVAL_DATASET_ROOT_DIR, x))
            if dataset_name in ['RSICD', 'RSITMD', 'ucmcaptions']:
                df[caption_key] = df[caption_key].apply(lambda x: 'a satellite image. ' + x)
            df_image = df.groupby('filepath').count().reset_index()
            df_text = df.groupby(caption_key).count().reset_index()
            dataset_image = CsvDataset_image(df=df_image, 
                                            transforms=preprocess_fn,
                                            img_key='filepath',
                                            return_img_path=True)
            dataloader_image = DataLoader(dataset_image, 
                                    batch_size=args.test_batch_size, 
                                    shuffle=False, 
                                    num_workers=args.workers*4)
            dataset_text = CsvDataset_text(df=df_text, 
                                            caption_key=caption_key, 
                                            tokenizer=tokenizer, 
                                            return_original_text=True)
            dataloader_text = DataLoader(dataset_text, 
                                        batch_size=args.test_batch_size, 
                                        shuffle=False, 
                                        num_workers=args.workers*4)
            retrieval_dataloader[dataset_name] = {'dataloader_image': dataloader_image, 
                                                  'dataloader_text': dataloader_text,
                                                  'batch_size': args.test_batch_size,
                                                  'df': df,
                                                  'caption_key': caption_key}
            
    return retrieval_dataloader

def test_zero_shot_retrieval(model, dataloaders, args, debugging=False):
    
    
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    dataloader_image = dataloaders['dataloader_image']
    dataloader_text = dataloaders['dataloader_text']
    batch_size = dataloaders['batch_size']
    df = dataloaders['df']
    caption_key = dataloaders['caption_key']
    
    model.eval()
    res = {}
    device = args.device
    all_image_features = []
    all_text_features = []
    all_image_paths = []
    all_text_original_texts = []
    with torch.no_grad():
        for batch in tqdm(dataloader_image, unit_scale=batch_size, desc='Extracting image features'):
            images, image_paths = batch
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            images = images.to(device)
            with autocast():
                if args.distributed and not args.horovod:
                    image_features = model.module.encode_image(images)
                else:
                    image_features = model.encode_image(images)
                all_image_features.append(image_features.cpu())
                all_image_paths.extend(image_paths)
        for batch in tqdm(dataloader_text, unit_scale=batch_size, desc='Extracting text features'):
            texts, original_texts = batch
            texts = texts.to(device)
            with autocast():
                if args.distributed and not args.horovod:
                    text_feature = model.module.encode_text(texts, normalize=True)
                else:
                    text_feature = model.encode_text(texts, normalize=True)
                all_text_features.append(text_feature.cpu())
                all_text_original_texts.extend(original_texts)
    with autocast():
        all_image_features = torch.cat(all_image_features, dim=0)
        all_text_features = torch.cat(all_text_features, dim=0)
        print(f'dtype befor normalization: {all_image_features.dtype}')
        all_image_features = F.normalize(all_image_features, dim=1).to(dtype=all_text_features.dtype)
        print(f'dtype after normalization: {all_image_features.dtype}')
    
    text_indices = {x: i for i, x in enumerate(all_text_original_texts)}
    img_indices = {x: i for i, x in enumerate(all_image_paths)}
    
    # ground truth
    img_path2text = {}
    text2img_path = {}
    for i in tqdm(df.index):
        text = df.loc[i, caption_key]
        img_path = df.loc[i, 'filepath']
        text_id = text_indices[text]
        img_id = img_indices[img_path]
        if img_path not in img_path2text:
            img_path2text[img_path] = set()
        img_path2text[img_path].add(text_id)
        if text not in text2img_path:
            text2img_path[text] = set()
        text2img_path[text].add(img_id)
        
    res = {'text2img_R@' + str(k): 0 for k in [1, 5, 10, 100]}
    res.update({'img2text_R@' + str(k): 0 for k in [1, 5, 10, 100]})
    
    # text to image
    logit_scale = 100.
    for i in tqdm(range(len(all_text_original_texts)), desc='Text to Image Retrieval'):
        text_feature = all_text_features[i]
        logits = logit_scale * text_feature @ all_image_features.t()
        ranking = torch.argsort(logits, descending=True).cpu().numpy()
        for k in [1, 5, 10, 100]:
            intersec = set(ranking[:k]) & set(text2img_path[all_text_original_texts[i]])
            if intersec:
                res['text2img_R@' + str(k)] += 1
    for k in [1, 5, 10, 100]:
        res['text2img_R@' + str(k)] /= len(all_text_original_texts)
        
    # image to text
    logit_scale = 100
    for i in tqdm(range(len(all_image_paths)), desc='Image to Text Retrieval'):
        image_feature = all_image_features[i]
        logits = logit_scale * image_feature @ all_text_features.t()
        ranking = torch.argsort(logits, descending=True).cpu().numpy()
        for k in [1, 5, 10, 100]:
            intersec = set(ranking[:k]) & img_path2text[all_image_paths[i]]
            if intersec:
                res['img2text_R@' + str(k)] += 1
    for k in [1, 5, 10, 100]:
        res['img2text_R@' + str(k)] /= len(all_image_paths)
        
    res['text2img_avg'] = sum([res['img2text_R@'+str(level)] for level in [1,5,10]]) / 3
    res['img2text_avg'] = sum([res['text2img_R@'+str(level)] for level in [1,5,10]]) / 3
    
    return res