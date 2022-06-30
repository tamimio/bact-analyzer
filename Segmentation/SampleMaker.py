from utils import *

import os
from pathlib import Path

import cv2
from PIL import Image
from matplotlib import pyplot as plt

def create_sample( image, images_for_grid ):
    ## 1 OTSU
    ret, res = cv2.threshold(image, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    images_for_grid.append(Image.fromarray(res.astype('uint8')))

    ## 2 THRESH_BINARY
    for val in [100, 110, 120, 130, 140, 150]:
        ret,res = cv2.threshold(image, val+10, 255, cv2.THRESH_BINARY)
        images_for_grid.append( Image.fromarray(res.astype('uint8')) )

    ## 3 adaptiveThreshold

    ### mean , gaussian
    param1 = 11
    param2 =  5

    # 11,2 ; 3,4

    for thr_type in [ cv2.ADAPTIVE_THRESH_MEAN_C, cv2.ADAPTIVE_THRESH_GAUSSIAN_C ]:

        res = cv2.adaptiveThreshold(image, 255, thr_type, cv2.THRESH_BINARY, param1-4,param2-2)
        images_for_grid.append( Image.fromarray(res.astype('uint8')) )
        res = cv2.adaptiveThreshold(image, 255, thr_type, cv2.THRESH_BINARY, param1,param2)
        images_for_grid.append( Image.fromarray(res.astype('uint8')) )
        res = cv2.adaptiveThreshold(image, 255, thr_type, cv2.THRESH_BINARY, param1+4,param2+2)
        images_for_grid.append( Image.fromarray(res.astype('uint8')) )

        res = cv2.adaptiveThreshold(image, 255, thr_type, cv2.THRESH_BINARY, param1-4,param2)
        images_for_grid.append( Image.fromarray(res.astype('uint8')) )
        res = cv2.adaptiveThreshold(image, 255, thr_type, cv2.THRESH_BINARY, param1-4,param2+2) #
        images_for_grid.append( Image.fromarray(res.astype('uint8')) )
        res = cv2.adaptiveThreshold(image, 255, thr_type, cv2.THRESH_BINARY, param1+4,param2)
        images_for_grid.append( Image.fromarray(res.astype('uint8')) )

        res = cv2.adaptiveThreshold(image, 255, thr_type, cv2.THRESH_BINARY, param1,param2-2)
        images_for_grid.append( Image.fromarray(res.astype('uint8')) )
        res = cv2.adaptiveThreshold(image, 255, thr_type, cv2.THRESH_BINARY, param1+4,param2-2)
        images_for_grid.append( Image.fromarray(res.astype('uint8')) )
        res = cv2.adaptiveThreshold(image, 255, thr_type, cv2.THRESH_BINARY, param1,param2+2)
        images_for_grid.append( Image.fromarray(res.astype('uint8')) )
        
    # create grid image
    grid = image_grid(images_for_grid, rows=9, cols=3)
    
    return grid

def save_sample ( directory, filename, sample_image ):
    filename_to_save = filename.split('.')[0]# + '_segm'
    sample_image.save(f'{directory}/{filename_to_save}.jpg')

def create_sample_for_image( full_filename, output_filename='' ):
    _, img, grid_sample = open_image( full_filename )

    grid_sample = create_sample( img, grid_sample )

    if output_filename == '':
        output_filename = '_'.join( full_filename.split('/')[-2:] )
        output_filename = output_filename.split('.')[0]
        output_filename = 'sample_'+output_filename

    save_sample( f'./result/', output_filename, grid_sample )

def create_sample_for_class( directory, class_name, _num ):
    print ( f'Creating samples for {class_name}...' )

    supdir = directory.split('/')[-1]
    directory = f'{directory}/{class_name}'

    # create resulting folder
    samples_folder_name = f"samples_{supdir}_{class_name.replace('/', '_')}"
    Path(f'./result/{samples_folder_name}').mkdir(parents=True, exist_ok=True)

    for file in os.listdir(directory):
        if _num == 0:
            break
        else:
            _num-=1
    
        filename = os.fsdecode(file)

        create_sample_for_image( f'{directory}/{filename}', f'{samples_folder_name}/{filename}' )

def create_sample_for_set( directory, set_type='train' ):
    print ( f'Creating samples for {set_type} set in {directory}...' )

    directory = f'{directory}/{set_type}'
    
    for subdir, dirs, files in os.walk(directory):
        for species in dirs:
            create_sample_for_class( directory, species, SAMPLES_COUNT )

