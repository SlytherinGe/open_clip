from torch.utils.data import Dataset
import os
import json
import sqlite3
import pandas as pd
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import logging

def _read_image_on_disk(image_collection, image_name, patch_name, ext, root):
    
    img_path = os.path.join(root, image_collection, image_name, patch_name + '.' + ext.lower())
    img = Image.open(img_path).convert('RGB')
    return img

class SQLiteDataset(Dataset):
    def __init__(self, annotation_db, 
                       meta_db, 
                       transforms, 
                       img_data_backend=dict(type='disk', root='.'), 
                       tokenizer=None):
        
        
        self.annotation_db = annotation_db
        self.meta_db = meta_db
        self.transforms = transforms
        self.img_data_backend = img_data_backend
        self.tokenizer = tokenizer
        self.img_read_fn = eval(f'_read_image_on_{self.img_data_backend["type"]}')
        self.read_kwargs = self.img_data_backend.copy()
        self.read_kwargs.pop('type')
        
        self.annotation_conn = sqlite3.connect(self.annotation_db)
        self.meta_conn = sqlite3.connect(self.meta_db)
        logging.info(f'annotation_db: {self.annotation_db}, meta_db: {self.meta_db}')
        # read all the annotation into pandas dataframe
        logging.info('Reading annotation data from sqlite')
        self.annotation_df = pd.read_sql_query("SELECT ID, PATCH, ANNOTATION FROM annotation", self.annotation_conn)
        self.annotation_df.rename(columns={'ANNOTATION': 'caption', 'PATCH': 'patch_id', 'ID': 'annotation_id'}, inplace=True)
        
        # the patch column stores the patch id, we use it to query the meta_db to get the image_info
        self.patch_meta_df = pd.read_sql_query("SELECT ID, NAME, IMAGE_NAME, FILE_FORMAT FROM patch", self.meta_conn)
        self.patch_meta_df.rename(columns={'ID': 'patch_id', 'NAME': 'patch_name', 'IMAGE_NAME': 'image_name', 'FILE_FORMAT': 'ext'}, inplace=True)
        self.image_meta_df = pd.read_sql_query("SELECT NAME, COLLECTION FROM image", self.meta_conn)
        self.image_meta_df.rename(columns={'NAME': 'image_name', 'COLLECTION': 'image_collection'}, inplace=True)
        logging.info('Reading annotation data from sqlite done')
        
        # merge the two dataframes
        logging.info('Merging annotation and meta data')
        self.df = pd.merge(self.annotation_df, self.patch_meta_df, on='patch_id', how='left')
        self.df = pd.merge(self.df, self.image_meta_df, on='image_name', how='left')
        logging.info('Merging annotation and meta data done')
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        read_kwargs = self.read_kwargs.copy()
        image_collection = row['image_collection']
        image_name = row['image_name']
        patch_name = row['patch_name']
        ext = row['ext']
        image_collection = '_'.join(image_collection.split('/'))
        read_kwargs.update(dict(image_collection=image_collection, image_name=image_name, patch_name=patch_name, ext=ext))
        img = self.img_read_fn(**read_kwargs)
        if self.transforms is not None:
            img = self.transforms(img)
        caption = row['caption']
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)
        return img, caption
        

# test the dataset
if __name__ == '__main__':
    from tqdm import tqdm
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print('loading dataset')
    
    dataset = SQLiteDataset(annotation_db='/mnt/FastDisk/GeJunYao/VLP/databases/backups/2024-3-19/annotation.db',
                            meta_db='/mnt/FastDisk/GeJunYao/VLP/databases/backups/2024-3-19/metadata.db',
                            transforms=transform,
                            img_data_backend=dict(type='disk', root='/mnt/SrvDataDisk/RSVLD'),
                            tokenizer=None)
    print('dataset loaded')
    for i in tqdm(range(len(dataset))):
        
        img, caption = dataset[i]
        
        