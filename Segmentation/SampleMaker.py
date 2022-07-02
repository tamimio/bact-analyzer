from utils import *

import os
from pathlib import Path

from PIL import Image

from Segmentator import Segmentator

class SampleMaker:
    res_path = './SampleMaker/'
        
    @classmethod
    def default_outFilename( cls, base_fullFilename ):
        output_filename = '_'.join( base_fullFilename.split('/')[-3:] )
        output_filename = f'{cls.res_path}sample_{output_filename}'
        return output_filename
        
    @classmethod
    def add_to_grid( cls, images, grid=[] ):
        for image in images:
            grid.append( Image.fromarray(image.astype('uint8')) )
        return grid
        
    @classmethod
    def create_sample( cls, img_orig, image, full=True ):
        grid_segm = []
        grid_segm = cls.add_to_grid( [img_orig, image], grid_segm )
        
        # segmentation methods
        
        samples = Segmentator.process( image, _apply_mask=False )
        grid_segm = cls.add_to_grid( samples, grid_segm )
        im_segm = cls.create_grid( grid_segm )
    
        if not full:
            return im_segm 
            
        else: # + apllying mask
            grid_masked=[]
            grid_masked = cls.add_to_grid( [img_orig, image], grid_masked )

            samples = Segmentator.process( image, _postprocess_mask=True, _apply_mask=True )
            grid_masked = cls.add_to_grid( samples, grid_masked )
            im_masked = cls.create_grid( grid_masked )
            
            return concat_h( im_segm, im_masked )
            
    @classmethod
    def create_grid( cls, grid=None ):
        _cols = 3
        _rows = len(grid) // 3
        return image_grid( grid, rows=_rows, cols=_cols )
        
    @classmethod
    def save_sample ( cls, filename, sample_image ):
        Path(cls.res_path).mkdir(parents=True, exist_ok=True)
        sample_image.save( filename )
        
    @classmethod
    def process_image( cls, full_filename, output_filename='', full_sample = True ):
        img_orig, img_bw = open_image( full_filename )
        
        sample = cls.create_sample( img_orig, img_bw, full=full_sample )
        
        if not output_filename: output_filename = cls.default_outFilename( full_filename )
        cls.save_sample( output_filename, sample )
