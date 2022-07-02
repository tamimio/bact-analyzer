# -*- coding: cp1251 -*-
print('Running do_segm.py')

from SampleMaker import SampleMaker
from DatasetSegmentator import DatasetSegmentator

DATASET_PATH  = '/media/DISK3/Bacteria_db/2021_2022/splitted/'
SAMPLES_COUNT = 15

### SAMPLES FOR 1 IMAGE

SampleMaker.process_image( DATASET_PATH+'train/Candida albicans/2.jpeg', full_sample=False )

### SAMPLES FOR 1 CLASS

segmentator = DatasetSegmentator( DATASET_PATH, samples_number = SAMPLES_COUNT )
segmentator.sample_class( 'Candida albicans' )

### SAMPLES FOR ALL CLASSES IN SET

#segmentator.sample_set( set_type='train' )

### SAMPLES FOR WHOLE DATASET

#segmentator.sample_dataset()

print('do_segm.py finished')
