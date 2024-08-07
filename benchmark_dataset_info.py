# Please replace this with the root directory of benchmark datasets
BENCHMARK_DATASET_ROOT_DIR = '{}'
RETRIEVAL_DATASET_ROOT_DIR = '/mnt/FastDisk/Datasets/RSCaptionDatasets'

BENCHMARK_DATASET_INFOMATION = {
    'aid': {
        'classification_mode': 'multiclass',
        'test_data': BENCHMARK_DATASET_ROOT_DIR + '/aid/aid_img_txt_pairs_test.csv',
        'classnames': BENCHMARK_DATASET_ROOT_DIR + '/aid/classnames.txt',
        'csv_separator': ',',
        'csv_img_key': 'filepath',
        'csv_class_key': 'label',
    },
    'eurosat': {
        'classification_mode': 'multiclass',
        'test_data': BENCHMARK_DATASET_ROOT_DIR + '/eurosat/eurosat_img_txt_pairs_test.csv',
        'classnames': BENCHMARK_DATASET_ROOT_DIR + '/eurosat/classnames.txt',
        'csv_separator': ',',
        'csv_img_key': 'filepath',
        'csv_class_key': 'label',
    },
    'fmow': {
        'classification_mode': 'multiclass',
        'test_data': BENCHMARK_DATASET_ROOT_DIR + '/fmow/fmow_img_txt_pairs_val.csv',
        'classnames': BENCHMARK_DATASET_ROOT_DIR + '/fmow/classnames.txt',
        'csv_separator': ',',
        'csv_img_key': 'filepath',
        'csv_class_key': 'label',
    },
    'nwpu': {
        'classification_mode': 'multiclass',
        'test_data': BENCHMARK_DATASET_ROOT_DIR + '/nwpu/img_txt_pairs_train.csv',
        'classnames': BENCHMARK_DATASET_ROOT_DIR + '/nwpu/classnames.txt',
        'csv_separator': ',',
        'csv_img_key': 'filepath',
        'csv_class_key': 'label',
    },
    'patternnet': {
        'classification_mode': 'multiclass',
        'test_data': BENCHMARK_DATASET_ROOT_DIR + '/patternnet/img_txt_pairs_train.csv',
        'classnames': BENCHMARK_DATASET_ROOT_DIR + '/patternnet/classnames.txt',
        'csv_separator': ',',
        'csv_img_key': 'filepath',
        'csv_class_key': 'label',
    },
    'SkyScript_cls': {
        'classification_mode': 'multiclass',
        'test_data': BENCHMARK_DATASET_ROOT_DIR + '/SkyScript_cls/img_txt_pairs_val.csv',
        'classnames': BENCHMARK_DATASET_ROOT_DIR + '/SkyScript_cls/classnames.txt',
        'csv_separator': ',',
        'csv_img_key': 'filepath',
        'csv_class_key': 'label',
    },
    'millionaid': {
        'classification_mode': 'multiclass',
        'test_data': BENCHMARK_DATASET_ROOT_DIR + '/millionaid/img_txt_pairs_train.csv',
        'classnames': BENCHMARK_DATASET_ROOT_DIR + '/millionaid/classnames.txt',
        'csv_separator': ',',
        'csv_img_key': 'filepath',
        'csv_class_key': 'label',
    },
    'rsicb': {
        'classification_mode': 'multiclass',
        'test_data': BENCHMARK_DATASET_ROOT_DIR + '/rsicb256/img_txt_pairs_train.csv',
        'classnames': BENCHMARK_DATASET_ROOT_DIR + '/rsicb256/classnames.txt',
        'csv_separator': ',',
        'csv_img_key': 'filepath',
        'csv_class_key': 'label',
    },
}


DATASET_CSV_FILE_PATH = {
    'SkyScript': f'{RETRIEVAL_DATASET_ROOT_DIR}/SkyScript_test_30K_filtered_by_CLIP_openai.csv',
    'RSICD': f'{RETRIEVAL_DATASET_ROOT_DIR}/RSICD/RSICD_img_txt_pairs_test.csv',
    'RSITMD': f'{RETRIEVAL_DATASET_ROOT_DIR}/RSITMD/RSITMD_img_txt_pairs_test.csv',
    'ucmcaptions': f'{RETRIEVAL_DATASET_ROOT_DIR}/ucmcaptions/ucmcaptions_img_txt_pairs_test.csv',
}