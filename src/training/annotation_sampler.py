import sqlite3
import pandas as pd
import logging
import webdataset as wds
import io


def random_sample(selected_dataframe: pd.DataFrame, **kwargs):
    
    return selected_dataframe.sample(n=1)
 
class AnnotationSampler:
    
    def __init__(self, annotation_db, sample_policy=dict(type='random')):
        self.annotation_db = annotation_db
        # load annotation database and cache it in dataframe
        self.conn = sqlite3.connect(self.annotation_db)
        logging.info(f"Loading annotation database from {self.annotation_db}")
        self.annotation_db = pd.read_sql_query("SELECT * FROM annotation", self.conn)
        logging.info(f"Annotation database loaded, shape: {self.annotation_db.shape}")
        self.annotation_db.set_index('ID', inplace=True)
        
        self.sample_policy = sample_policy.copy()
        self.sample_type = self.sample_policy.pop('type')
        self.sample_func = eval(f"{self.sample_type}_sample")
        
        
    def __call__(self, patch_id: int):
        
        selected_dataframe = self.annotation_db[self.annotation_db['PATCH']==patch_id]
        if selected_dataframe.empty:
            return None
        selected_row = self.sample_func(selected_dataframe, **self.sample_policy)
        return selected_row['ANNOTATION'].values[0]
    
class WDSAnnotationSampler(wds.PipelineStage):
    
    def __init__(self, annotation_db: str, sample_policy: dict=dict(type='random'), include_dataset: str='RSVLD'):
        
        self.sampler = AnnotationSampler(annotation_db, sample_policy)
        self.include_dataset = include_dataset
        
    def run(self, stream):
        
        for sample in stream:
            
            if self.include_dataset not in sample['__url__']:
                yield sample
                continue
            
            patch_id = int(sample['__key__'])
            annotation = self.sampler(patch_id)
            if annotation is not None:
                sample['txt'] = annotation.encode('utf-8')
            yield sample

# test
if __name__ == '__main__':
    sampler = AnnotationSampler('/mnt/SrvUserDisk/Gejunyao/VLP/test_downloader/annotation.db', sample_policy=dict(type='random'))
    print(sampler(162))
    print(sampler(162))
    print(sampler(162))
    print(sampler(162))
    print(sampler(162))
    print(sampler(162))
    print(sampler(162))