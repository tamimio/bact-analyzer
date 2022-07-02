from SampleMaker import SampleMaker
from Segmentator import Segmentator

import os
from pathlib import Path

class DatasetSegmentator:

    def __init__( self, db_path, res_path = './segmented/', sample_path = './samples/', samples_number = 15 ):
        self.db_path        = db_path
        self.res_path       = res_path
        self.sample_path    = sample_path
        self.samples_number = samples_number
        
        Path( self.res_path    ).mkdir(parents=True, exist_ok=True)
        Path( self.sample_path ).mkdir(parents=True, exist_ok=True)
        
        print(f'[DatasetSegmentator] Source path: {self.db_path}')
        print(f'[DatasetSegmentator] Result path: {self.res_path}')
        print(f'[DatasetSegmentator] Samples path: {self.sample_path}; Number of samples: {self.samples_number}')

    def __del__( self ):
        if len( os.listdir(self.res_path   ) ) == 0: os.rmdir( self.res_path )
        if len( os.listdir(self.sample_path) ) == 0: os.rmdir( self.sample_path )

    # SAMPLING

    def sample_class( self, class_name, set_type='train' ):
        print ( f'[DatasetSegmentator] Creating samples for {set_type} set of {class_name}...' )
        src_directory = f'{self.db_path}{set_type}/{class_name}'
        
        # create folder
        samples_folder_name = f"{self.sample_path}{set_type}_{class_name}"
        Path(f'{samples_folder_name}').mkdir(parents=True, exist_ok=True)
        
        _num = self.samples_number

        for file in os.listdir(src_directory):
            if _num == 0:
                break
            else:
                _num-=1
        
            filename = os.fsdecode(file)

            SampleMaker.process_image( full_filename   = f'{src_directory}/{filename}',
                                       output_filename = f'{samples_folder_name}/{filename}',
                                       full_sample     = True)

    def sample_set( self, set_type='train' ):
        directory = self.db_path + set_type
        
        for subdir, dirs, files in os.walk(directory):
            for species in dirs:
                self.sample_class( species, set_type )

    def sample_dataset( self ):
        for set_type in [ 'test', 'train' ]:
            self.sample_set( set_type=set_type )

    # FULL PROCESSING

    def process_class( self, class_name, set_type='train' ):
        pass
        
    def process_set( self, set_type='train' ):
        pass
        
    def process_dataset( self ):
        pass