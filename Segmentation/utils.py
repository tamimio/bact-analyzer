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
