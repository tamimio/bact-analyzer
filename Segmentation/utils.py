import cv2
from PIL import Image
from matplotlib import pyplot as plt

def open_image( filename ):
	img_orig = plt.imread( filename ) 
	img_bw = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

	return img_orig, img_bw

def image_grid( imgs, rows, cols ):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def create_range( start, number, step ):
    return range(start, start+step*number, step)
