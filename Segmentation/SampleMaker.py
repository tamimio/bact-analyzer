from utils import *

import os
from pathlib import Path

from PIL import Image

from Segmentator import Segmentator

class SampleMaker:
    
    def __init__(self, db_path, res_path = './samples/', samples_number = 15 ):
        self.db_path  = db_path
        self.res_path = res_path
        self.samples_number = samples_number
        
        Path(f'{self.res_path}').mkdir(parents=True, exist_ok=True)
        
        print(f'Source path: {self.db_path}')
        print(f'Result path: {self.res_path}')
        print(f'Number of samples: {self.samples_number}')
        print('SampleMaker is ready.')
        
    def default_outFilename( self, base_fullFilename ):
        output_filename = '_'.join( base_fullFilename.split('/')[-2:] )
        output_filename = output_filename.split('.')[0]
        output_filename = 'sample_'+output_filename
        return output_filename
        
    def add_to_grid( self, images, grid=[] ):
        for image in images:
            grid.append( Image.fromarray(image.astype('uint8')) )
        return grid

    def create_sample( self, img_orig, image, full=True ):
        grid_segm = []
        grid_segm = self.add_to_grid( [img_orig, image], grid_segm )
        
        # segmentation methods
        
        samples = Segmentator.process( image, _postprocess_mask=False, _apply_mask=False )
        grid_segm = self.add_to_grid( samples, grid_segm )
        grid_segm = self.create_grid( grid_segm )
    
        if not full:
            return self.create_grid( grid_segm )
            
        else: # + apllying mask
            grid_masked=[]
            grid_masked = self.add_to_grid( [img_orig, image], grid_masked )

            samples = Segmentator.process( image, _postprocess_mask=True, _apply_mask=True )
            grid_masked = self.add_to_grid( samples, grid_masked )
            grid_masked = self.create_grid( grid_masked )
            
            return concat_h( grid_segm, grid_masked )

    def create_grid( self, grid=None ):
        if not grid: grid=self.grid
        _cols = 3
        _rows = len(grid) // 3
        return image_grid( grid, rows=_rows, cols=_cols )

    def save_sample ( self, directory, filename, sample_image ):
        filename_to_save = filename.split('.')[0]# + '_segm'
        sample_image.save(f'{directory}/{filename_to_save}.jpg')
    
    def process_image( self, full_filename, output_filename='' ):
        img_orig, img_bw = open_image( full_filename )
        
        sample = self.create_sample( img_orig, img_bw, full=True )
        
        if not output_filename: output_filename = self.default_outFilename( full_filename )
        self.save_sample( self.res_path, output_filename, sample )

    def process_class( self, class_name, set_type='train' ):
        print ( f'Creating samples for {class_name}...' )

        directory = f'{self.db_path}{set_type}/{class_name}'

        # create resulting folder
        samples_folder_name = f"samples_{set_type}_{class_name.replace('/', '_')}"
        Path(f'{self.res_path}{samples_folder_name}').mkdir(parents=True, exist_ok=True)

        _num = self.samples_number

        for file in os.listdir(directory):
            if _num == 0:
                break
            else:
                _num-=1
        
            filename = os.fsdecode(file)

            self.process_image( f'{directory}/{filename}', f'{samples_folder_name}/{filename}' )

    def process_set( self, set_type='train' ):
        print ( f'Creating samples for {set_type} set in {self.db_path}...' )

        directory = self.db_path + set_type
        
        for subdir, dirs, files in os.walk(directory):
            for species in dirs:
                self.process_class( species, set_type )

