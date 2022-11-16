# -*- coding: cp1251 -*-
print('Running do_segm.py')

from SampleMaker import SampleMaker
from DatasetSegmentator import DatasetSegmentator

DATASET_PATH  = '/media/DISK3/Bacteria_db/2021_2022/splitted/'
SAMPLES_COUNT = 15

### SAMPLES FOR 1 IMAGE

# SampleMaker.process_image( DATASET_PATH+'train/Citrobacter braakii/1687.jpg', full_sample=True )

### SAMPLES FOR 1 CLASS

segmentator = DatasetSegmentator( DATASET_PATH, 
                                  mask_path = '/media/DISK3/Bacteria_db/2021_2022/masks/',     # was 'segmented'
                                  segm_path = '/media/DISK3/Bacteria_db/2021_2022/segmented/',
                                  samples_number = SAMPLES_COUNT )

# segmentator.sample_class( 'Citrobacter braakii' )

### SAMPLES FOR ALL CLASSES IN SET

# segmentator.sample_set( set_type='train' ) # done

### SAMPLES FOR WHOLE DATASET

#segmentator.sample_dataset()


### - SEGMENT CLASS -

segmentator.process_class ( 'Citrobacter braakii', 'train' )
# segmentator.apply_mask()

### --- SEGMENT WHOLE DATASET ---

# segmentator.process_dataset()


print('do_segm.py finished')
