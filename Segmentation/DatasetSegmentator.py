from SampleMaker import SampleMaker
from Segmentator import Segmentator

import os
import numpy as np
from pathlib import Path
from PIL import Image
import json

from utils import open_image

class DatasetSegmentator:

    def __init__( self, db_path,                                 # full path to dataset
                        mask_path      = './masks/',             # resulting path with masks (after full class processing)
                        segm_path      = './segmented/',         # resulting path with segmented images (after full class proc.) - disabled
                        sample_path    = './samples/',           # resulting path with samples (v. SampleMaker)
                        samples_number = 15,                     # numer of samples to make with SampleMaker
                        rules_filename = 'BacteriaDB_rules.json' # full filename of JSON file with segmenting rules
                        ):
        self.db_path        = db_path
        self.mask_path      = mask_path
        self.sample_path    = sample_path
        self.samples_number = samples_number
        self.segm_path      = segm_path
        self.rules_filename = rules_filename
        
        Path( self.mask_path    ).mkdir(parents=True, exist_ok=True)
        Path( self.sample_path ).mkdir(parents=True, exist_ok=True)
        
        print(f'[DatasetSegmentator] Source path:  {self.db_path}')
        print(f'[DatasetSegmentator] Masks path:   {self.mask_path}')
        print(f'[DatasetSegmentator] Rules:        {self.rules_filename}')
        print(f'[DatasetSegmentator] Samples path: {self.sample_path}; Number of samples: {self.samples_number}')

    def __del__( self ):
        if len( os.listdir(self.mask_path   ) ) == 0: os.rmdir( self.mask_path )
        if len( os.listdir(self.sample_path) ) == 0: os.rmdir( self.sample_path )

    ### --- UTILS ---
    
    def load_rules( self, filename, class_name ):
        f = open(self.rules_filename)
        rules = json.load(f)
        f.close()
        
        ok = class_name in rules
        
        return rules, ok

    ### --- SAMPLING ---

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
            print(dirs)
            for species in dirs:
                self.sample_class( species, set_type )

    def sample_dataset( self ):
        for set_type in [ 'test', 'train' ]:
            self.sample_set( set_type=set_type )

    ### --- FULL PROCESSING (CREATE MASK) ---
    
    def process_image_by_class_rules( self, image, rule ): # create mask
        
        if rule["method"] == 'Adaptive':
            p_method     = rule["params"][0]
            p_blockSize  = rule["params"][1]
            p_subtr      = rule["params"][2]
            postproc_acc = rule["postproc_acc"]
            
            mask = Segmentator.segm_Adaptive( image, _method=p_method, _block_size=p_blockSize, _subtr=p_subtr )
            
            if postproc_acc != 0:
                mask = Segmentator.postproc_mask_1 ( mask, _acc=postproc_acc )
                
        #    #segm = Segmentator.apply_mask( image, mask )
        
        mask = Segmentator.invert_mask( mask )
            
        return mask#, segm

    def process_class( self, class_name, set_type='train' ):
        # print(f'[DatasetSegmentator] process_class {class_name}')
        
        # set directories
        src_directory  = f'{self.db_path}{set_type}/{class_name}'
        mask_directory = f'{self.mask_path}{set_type}/{class_name}'
        segm_directory = f'{self.segm_path}{set_type}/{class_name}'
        
        # load rules
        rules, ok = self.load_rules('BacteriaDB_rules.json', class_name)
        
        if not ok:
            print(f'[DatasetSegmentator] Cannot process class {class_name}, no rules found')
            return
        else:
            print(f'[DatasetSegmentator] Rules for {class_name}: {rules[class_name]}')
        
        # create folders
        Path(f'{mask_directory}').mkdir(parents=True, exist_ok=True)
        
        # process each image of the class
        for file in os.listdir(src_directory):
            
            # prepare src dst files
            filename = os.fsdecode(file)
            src_filename = f'{src_directory}/{filename}'
            dst_filename = f'{mask_directory}/mask_{filename}'
            # print( f'src_filename {src_filename}' )
            # print( f'dst_filename {dst_filename}' )
            
            # open image
            img_orig, img_bw = open_image( src_filename )
            
            # process image
            mask = self.process_image_by_class_rules( img_bw, rules[class_name] )
            
            # save mask
            mask = Image.fromarray(np.uint8(mask)).convert('RGB')
            mask.save( dst_filename )
            
            if False:
                Path(f'{segm_directory}').mkdir(parents=True, exist_ok=True)
                segm_filename = f'{segm_directory}/segm_{filename}'
                print( segm_filename )                
                segm = Image.fromarray(np.uint8(segm)).convert('RGB')
                segm.save( segm_filename )
            
        
    def process_set( self, set_type='train' ):
        print(f'[DatasetSegmentator] process_set {set_type}')
        directory = self.db_path + set_type
        
        for subdir, dirs, files in os.walk(directory):
            for species in dirs:
                self.process_class( species, set_type )
        
    def process_dataset( self ):
        print('[DatasetSegmentator] process_dataset')
        for set_type in [ 'test', 'train' ]:
            self.process_set( set_type=set_type )
            
            
    def apply_mask( self ): # disabled
        print('[DatasetSegmentator] apply_mask')
        for set_type in [ 'test', 'train' ]:
            directory = self.db_path + set_type
            
            for subdir, dirs, files in os.walk(directory):
                for class_name in dirs:
                    src_directory  = f'{self.db_path}{set_type}/{class_name}'
                    mask_directory = f'{self.mask_path}{set_type}/{class_name}'
                    
                    # create folder
                    segm_directory = f"{self.segm_path}{set_type}/{class_name}"
                    Path(f'{segm_directory}').mkdir(parents=True, exist_ok=True)
                    
                    for file in os.listdir(src_directory):
                        
                        filename = os.fsdecode(file)
                        img_filename  = f'{src_directory}/{filename}'
                        mask_filename = f'{mask_directory}/mask_{filename}'
                        segm_filename = f'{segm_directory}/segm_{filename}'
                        
                        if not Path(mask_filename).exists():
                            continue
                        
                        image, _ = open_image( img_filename )
                        mask, _ = open_image( mask_filename )
                        
                        print(img_filename)
                        print(mask_filename)
                        print(segm_filename)
                        
                        segm = Segmentator.apply_mask( image, mask )
                        segm = Image.fromarray(np.uint8(segm)).convert('RGB')
                        segm.save( segm_filename )