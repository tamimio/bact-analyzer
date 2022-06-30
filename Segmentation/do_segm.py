# -*- coding: cp1251 -*-
print('Running do_segm.py')

import os
from pathlib import Path

import cv2
from PIL import Image
from matplotlib import pyplot as plt


def open_image( filename ):
    img_orig = plt.imread( filename ) 
    img_bw = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

    images_for_grid = [Image.fromarray(img_orig.astype('uint8')), Image.fromarray(img_bw.astype('uint8'))]
    
    return img_orig, img_bw, images_for_grid

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def create_sample( image, images_for_grid ):
    ## 1 OTSU
    ret, res = cv2.threshold(image, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    res      = Image.fromarray(res.astype('uint8'))
    images_for_grid.append(res)

    ## 2 THRESH_BINARY
    for val in [100, 110, 120, 130, 140, 150]:
        ret,res = cv2.threshold(image, val+10, 255, cv2.THRESH_BINARY)
        images_for_grid.append( Image.fromarray(res.astype('uint8')) )

    ## 3 adaptiveThreshold

    ### mean , gaussian
    param1 = 11
    param2 =  5

    # 11,5 ; 3,4

    for thr_type in [ cv2.ADAPTIVE_THRESH_MEAN_C, cv2.ADAPTIVE_THRESH_GAUSSIAN_C ]:

        res = cv2.adaptiveThreshold(image, 255, thr_type, cv2.THRESH_BINARY, param1-4,param2-2)
        images_for_grid.append( Image.fromarray(res.astype('uint8')) )
        res = cv2.adaptiveThreshold(image, 255, thr_type, cv2.THRESH_BINARY, param1,param2)
        images_for_grid.append( Image.fromarray(res.astype('uint8')) )
        res = cv2.adaptiveThreshold(image, 255, thr_type, cv2.THRESH_BINARY, param1+4,param2+2)
        images_for_grid.append( Image.fromarray(res.astype('uint8')) )

        res = cv2.adaptiveThreshold(image, 255, thr_type, cv2.THRESH_BINARY, param1-4,param2)
        images_for_grid.append( Image.fromarray(res.astype('uint8')) )
        res = cv2.adaptiveThreshold(image, 255, thr_type, cv2.THRESH_BINARY, param1-4,param2+2)
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



DATASET_PATH  = '/media/DISK3/Bacteria_db/2021_2022/splitted'
SAMPLES_COUNT = 15

### SAMPLES FOR 1 IMAGE

create_sample_for_image( f'{DATASET_PATH}/train/Candida albicans/1.jpeg' )

### SAMPLES FOR 1 CLASS

create_sample_for_class( f'{DATASET_PATH}/train', 'Acinetobacter baumannii', SAMPLES_COUNT )

### SAMPLES FOR ALL CLASSES IN SET

# create_sample_for_set( DATASET_PATH, 'train' )

### SAMPLES FOR WHOLE DATASET

# for set_type in ['train', 'test']:
#     create_sample_for_set( DATASET_PATH, set_type )

print('do_segm.py finished')