from utils import *

import os
from pathlib import Path

from PIL import Image

from Segmentator import Segmentator

class SampleMaker:
	grid = []
    
	def __init__(self, db_path, res_path = './samples/', samples_number = 15 ):
		self.db_path  = db_path
		self.res_path = res_path
		self.samples_number = samples_number
        
		Path(f'{self.res_path}').mkdir(parents=True, exist_ok=True)
        
		print(f'Source path: {self.db_path}')
		print(f'Result path: {self.res_path}')
		print(f'Number of samples: {self.samples_number}')
		print('SampleMaker is ready.')
		
	def add_to_grid( self, images ):
		for image in images:
			self.grid.append( Image.fromarray(image.astype('uint8')) )

	def create_sample( self, image ):
		results = Segmentator.process( image )
		self.add_to_grid( results )

	def save_sample ( self, directory, filename, sample_image ):
		filename_to_save = filename.split('.')[0]# + '_segm'
		sample_image.save(f'{directory}/{filename_to_save}.jpg')
	
	def process_image( self, full_filename, output_filename='' ):
		img_orig, img = open_image( full_filename )
        
		self.add_to_grid( [img_orig, img] )
        
		self.create_sample( img )
        
		_cols = 3
		_rows = len(self.grid) // 3
		self.grid = image_grid(self.grid, rows=_rows, cols=_cols)

		if output_filename == '':
			output_filename = '_'.join( full_filename.split('/')[-2:] )
			output_filename = output_filename.split('.')[0]
			output_filename = 'sample_'+output_filename

		self.save_sample( self.res_path, output_filename, self.grid )

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

