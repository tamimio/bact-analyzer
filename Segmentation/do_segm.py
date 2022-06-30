# -*- coding: cp1251 -*-
print('Running do_segm.py')

from SampleMaker import SampleMaker

DATASET_PATH  = '/media/DISK3/Bacteria_db/2021_2022/splitted/'
SAMPLES_COUNT = 15

sample_maker = SampleMaker( DATASET_PATH, samples_number = SAMPLES_COUNT )

### SAMPLES FOR 1 IMAGE

sample_maker.process_image( DATASET_PATH+'train/Candida albicans/3.jpeg' )

### SAMPLES FOR 1 CLASS

sample_maker.process_class( class_name='Acinetobacter baumannii' )

### SAMPLES FOR ALL CLASSES IN SET

#sample_maker.process_set( set_type='train' )

### SAMPLES FOR WHOLE DATASET

#for set_type in ['train', 'test']:
#    sample_maker.process_set( set_type=set_type )

print('do_segm.py finished')
