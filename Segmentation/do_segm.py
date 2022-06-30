# -*- coding: cp1251 -*-
print('Running do_segm.py')

from SampleMaker import *

DATASET_PATH  = '/media/DISK3/Bacteria_db/2021_2022/splitted'
SAMPLES_COUNT = 15

### SAMPLES FOR 1 IMAGE

create_sample_for_image( f'{DATASET_PATH}/train/Candida albicans/3.jpeg' )

### SAMPLES FOR 1 CLASS

create_sample_for_class( f'{DATASET_PATH}/test', 'Acinetobacter baumannii', SAMPLES_COUNT )

### SAMPLES FOR ALL CLASSES IN SET

# create_sample_for_set( DATASET_PATH, 'train' )

### SAMPLES FOR WHOLE DATASET

# for set_type in ['train', 'test']:
#     create_sample_for_set( DATASET_PATH, set_type )

print('do_segm.py finished')