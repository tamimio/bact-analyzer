# -*- coding: cp1251 -*-
print('Running do_segm.py')

import os
from pathlib import Path

import cv2
from PIL import Image
from matplotlib import pyplot as plt

'''
# -d <directory> - segment all files in directory, save in result/<directory>_segm_method
# -m <method> - segment directory with specific method

# -f <filename>  - segment 1 file, save as grid in result/<filename>_segm
'''

DB_PATH  = '/media/DISK3/Bacteria_db/2021_2022/splitted/'
FILENAME = 'train/Acinetobacter baumannii/Снимок-4759.jpg'

def open_image(filename):
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

def create_sample(image, images_for_grid):
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

def create_sample_for_set(set_type='train'):
    print('create_sample_for_set')


species_folder = 'train/Acinetobacter baumannii'
directory_name = f'/media/DISK3/Bacteria_db/2021_2022/splitted/{species_folder}'
directory = os.fsencode(directory_name)

# create folder
folder_name = species_folder.replace('/', '_')
Path(f'./result/samples_{folder_name}').mkdir(parents=True, exist_ok=True)

SAMPLES_COUNT = 15

for subdir, dirs, files in os.walk('/media/DISK3/Bacteria_db/2021_2022/splitted/train'):
    
    for dir in dirs:
        print ( os.path.join(subdir, dir) )

for file in os.listdir(directory):
    filename = os.fsdecode(file)
        
    # OPEN IMAGE
    
    _, img, grid_sample = open_image( f'{directory_name}/{filename}' )
    
    # img_orig = plt.imread(f'{directory_name}/{filename}') #(f'{DB_PATH}{FILENAME}')
    # img_bw = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

    # images_for_grid = [Image.fromarray(img_orig.astype('uint8')), Image.fromarray(img_bw.astype('uint8'))]

    # SEGMENTATION 1-N

    grid_sample = create_sample( img, grid_sample )

    # SAVE
    
    # save image
    filename_to_save = filename.split('.')[0]    # filename_to_save = FILENAME.split('.')[0]
    # # filename_to_save = filename_to_save.replace('/', '_')
    grid_sample.save(f'./result/samples_{folder_name}/{filename_to_save}_segm.jpg')
    
    SAMPLES_COUNT -= 1
    if SAMPLES_COUNT == 0:
        break

print('do_segm.py finished')