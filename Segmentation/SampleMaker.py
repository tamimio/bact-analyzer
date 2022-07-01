from utils import *

import os
from pathlib import Path

import cv2
from PIL import Image
from matplotlib import pyplot as plt

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
		
	def add_to_grid( self, image ):
		self.grid.append( Image.fromarray(image.astype('uint8')) )
        
	def create_sample( self, image ):
		## 1 OTSU
		ret, res = cv2.threshold(image, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		self.add_to_grid( res )

		## 2 THRESH_BINARY
		for val in [100, 110, 120, 130, 140, 150]:
			ret,res = cv2.threshold(image, val+10, 255, cv2.THRESH_BINARY)
			self.add_to_grid( res )

		## 3 adaptiveThreshold

		### mean , gaussian
		param1 = 11
		param2 =  5

		# 11,2 ; 3,4

		for thr_type in [ cv2.ADAPTIVE_THRESH_MEAN_C, cv2.ADAPTIVE_THRESH_GAUSSIAN_C ]:

			res = cv2.adaptiveThreshold(image, 255, thr_type, cv2.THRESH_BINARY, param1-4,param2-2)
			self.add_to_grid( res )
			res = cv2.adaptiveThreshold(image, 255, thr_type, cv2.THRESH_BINARY, param1,param2)
			self.add_to_grid( res )
			res = cv2.adaptiveThreshold(image, 255, thr_type, cv2.THRESH_BINARY, param1+4,param2+2)
			self.add_to_grid( res )

			res = cv2.adaptiveThreshold(image, 255, thr_type, cv2.THRESH_BINARY, param1-4,param2)
			self.add_to_grid( res )
			res = cv2.adaptiveThreshold(image, 255, thr_type, cv2.THRESH_BINARY, param1-4,param2+2) #
			self.add_to_grid( res )
			res = cv2.adaptiveThreshold(image, 255, thr_type, cv2.THRESH_BINARY, param1+4,param2)
			self.add_to_grid( res )

			res = cv2.adaptiveThreshold(image, 255, thr_type, cv2.THRESH_BINARY, param1,param2-2)
			self.add_to_grid( res )
			res = cv2.adaptiveThreshold(image, 255, thr_type, cv2.THRESH_BINARY, param1+4,param2-2)
			self.add_to_grid( res )
			res = cv2.adaptiveThreshold(image, 255, thr_type, cv2.THRESH_BINARY, param1,param2+2)
			self.add_to_grid( res )
			

	def save_sample ( self, directory, filename, sample_image ):
		filename_to_save = filename.split('.')[0]# + '_segm'
		sample_image.save(f'{directory}/{filename_to_save}.jpg')
	
	def process_image( self, full_filename, output_filename='' ):
		img_orig, img = open_image( full_filename )
        
		self.add_to_grid( img_orig )
		self.add_to_grid( img )
        
		self.create_sample( img )
		self.grid = image_grid(self.grid, rows=9, cols=3)

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

