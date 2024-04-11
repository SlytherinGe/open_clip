from torch.utils.data import Dataset
import os
import json
import sqlite3
import pandas as pd
import numpy as np
import torch
from PIL import Image, ImageFile
import torchvision.transforms as transforms
import logging
from pymongo import MongoClient
import pymongo
import io
from tqdm import tqdm

def _read_image_on_disk_init(img_data_backend, read_kwargs):
    return read_kwargs

def _read_image_on_mongo_init(img_data_backend, read_kwargs):
    mongo_uri = img_data_backend['mongo_uri']
    mongo_client = MongoClient(mongo_uri)
    collection_buf = {}
    # iterate through all the collections and create indexes for "name"
    datasets = mongo_client.list_database_names()
    for dataset in datasets:
        # except default
        if dataset not in ['admin', 'config', 'local']:
            collection_buf[f'{dataset}'] = {}
            # iterate through all the collections
            collections = mongo_client[dataset].list_collection_names()
            for collection in tqdm(collections, desc=f'Creating indexes for {dataset}'):
                mongo_client[dataset][collection].create_index([('name', 1)], unique=True)
                collection_buf[f'{dataset}'][f'{collection}'] = mongo_client[dataset][collection]
    
    read_kwargs.pop('mongo_uri')
    read_kwargs.update(dict(mongo_client=mongo_client, collection_buf=collection_buf))
    return read_kwargs

def _read_image_on_disk(image_collection, image_name, patch_name, ext, root, **kwargs):
    img_path = os.path.join(root, image_collection, image_name, patch_name + '.' + ext.lower())
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        logging.warning(f'Failed to read image {img_path}. Error: {e}')
        if e == 'OSError':
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            img = Image.open(img_path).convert('RGB')
        else:
            raise e
    return img

def _read_image_on_mongo(image_collection, image_name, patch_name, ext, mongo_client, collection_buf, **kwargs):
    # this may be confusing, but the image_collection is actually the database name in mongo
    # and the image_name is the collection name
    mongo_collection = collection_buf[f'{image_collection}'][f'{image_name}']
    # mongo_db = mongo_client[image_collection]
    # mongo_collection = mongo_db[image_name]
    img_data = mongo_collection.find_one({'name': patch_name})
    try:
        img = Image.open(io.BytesIO(img_data['patch'])).convert('RGB')
    except Exception as e:
        logging.warning(f'Failed to read image {image_collection}/{image_name}/{patch_name}.{ext}. Error: {e}')
        if e == 'OSError':
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            img = Image.open(io.BytesIO(img_data['patch'])).convert('RGB')
        else:
            raise e
    return img

class SQLiteDataset(Dataset):
    def __init__(self, annotation_db, 
                       meta_db, 
                       transforms, 
                       img_data_backend=dict(type='disk', root='.'), 
                       tokenizer=None):
        super().__init__()
        self.annotation_db = annotation_db
        self.meta_db = meta_db
        self.transforms = transforms
        self.img_data_backend = img_data_backend
        self.tokenizer = tokenizer
        self.read_kwargs = self.img_data_backend.copy()
        self.backendtype = self.read_kwargs.pop('type')
        self.img_read_fn = eval(f'_read_image_on_{self.backendtype}')
        self.read_kwargs = eval(f'_read_image_on_{self.backendtype}_init')(self.img_data_backend, self.read_kwargs)
               
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
        
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
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
            caption = self.tokenizer([str(caption)])[0]
        return img, caption
        
    # disconnect the database connections
    def __del__(self):
        self.annotation_conn.close()
        self.meta_conn.close()

# test the dataset
if __name__ == '__main__':
    from tqdm import tqdm
    # import dataloader
    from torch.utils.data import DataLoader
    
    # define the transform
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
                            # img_data_backend=dict(type='mongo', mongo_uri='mongodb://localhost:27017'),
                            tokenizer=None)
    
    # test the throuput of querying a very large dictionary
    # collection_buf = dataset.read_kwargs['collection_buf']
    # keys_database = list(collection_buf.keys())
    # keys_collection = list(collection_buf[keys_database[0]].keys())
    # # shuffling the keys to test the random access
    # np.random.shuffle(keys_collection)
    # for key in tqdm(keys_collection):
    #     collection = collection_buf[keys_database[0]][key]
    #     cursor = collection.find_one()
    
    # test the througput of the dataset
    for i in tqdm(range(len(dataset))):
        img, caption = dataset[i]
    
    # # define the dataloader
    # dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=2)
    # # test the througput
    # for i, (img, caption) in tqdm(enumerate(dataloader)):
    #     pass