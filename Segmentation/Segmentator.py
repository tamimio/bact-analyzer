from utils import create_range

import cv2
import numpy as np
from PIL import Image, ImageFilter

class Segmentator:
    @classmethod
    def is_bw( cls, image ):
        shape = image.shape
        if ( len(shape)==3 and shape[-1]==3 ):
            return False
        return True

    @classmethod
    def prepare_image( cls, image ):
        if not cls.is_bw(image):
            return  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
        
    @classmethod
    def segm_Otsu( cls, image ):
        image = cls.prepare_image( image )
        ret, res = cv2.threshold(image, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return res

    @classmethod
    def segm_Binary( cls, image, thres_val=100 ):
        image = cls.prepare_image( image )
        ret,res = cv2.threshold(image, thres_val, 255, cv2.THRESH_BINARY)
        return res

    @classmethod
    def segm_Adaptive( cls, image, _method='mean', _block_size=11, _subtr=5 ):
        image = cls.prepare_image( image )
        
        method = cv2.ADAPTIVE_THRESH_MEAN_C if _method=='mean' else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        if _block_size % 2 != 1: _block_size +=1
        
        res = cv2.adaptiveThreshold(image, 255, method, cv2.THRESH_BINARY, _block_size, _subtr)
        return res

    @classmethod
    def postproc_mask_1( cls, image, _acc=0.02 ): # base: https://www.askpython.com/python/examples/image-segmentation
        kernel = np.ones((2, 2), np.uint8)
        closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE,kernel, iterations = 1)
        bg = cv2.dilate(closing, kernel, iterations = 1)
        dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)
        ret, res = cv2.threshold(dist_transform, _acc*dist_transform.max(), 255, 0)
        return res.astype(np.uint8)

    @classmethod
    def apply_mask( cls, image, mask_inv ):
        return cv2.bitwise_and( image, image, mask = cv2.bitwise_not(mask_inv) )

    @classmethod
    def process( cls, image, _postprocess_mask=False, _apply_mask=False ):
        image = cls.prepare_image( image )
        
        all_results = []
        
        ## 1 OTSU
        res = cls.segm_Otsu( image )
        if _postprocess_mask: res = cls.postproc_mask_1( res )
        if _apply_mask:       res = cls.apply_mask( image, res )
        all_results.append( res )

        ## 2 THRESH_BINARY
        range_binary = create_range( start=80, number=6, step=20 )
        for val in range_binary:
            res = cls.segm_Binary( image, thres_val=val )
            if _postprocess_mask: res = cls.postproc_mask_1( res )
            if _apply_mask:       res = cls.apply_mask( image, res )
            all_results.append( res )

        ## 3 ADAPTIVE_THRESHOLD
        for method in [ 'mean', 'gaussian' ]:
        
            range_blockSize = create_range( start=5, number=3, step=10 )
            range_param2 = create_range( start=3, number=3, step=2 )
            
            for param1 in range_blockSize:
                for param2 in range_param2:
                    res = cls.segm_Adaptive( image, _method=method, _block_size=param1, _subtr=param2 )
                    
                    res = Image.fromarray(res.astype('uint8'))
                    
                    res = res.filter(ImageFilter.GaussianBlur(1))
                    res = np.array(res)
                    #print( type(res) )
                    #res = cls.segm_Otsu( res )
                    #res = cls.segm_Binary( res, thres_val=100 )
                    #if _postprocess_mask:
                    #res = cls.postproc_mask_1( res )
                    #if _apply_mask:       
                    res = cls.apply_mask( image, res )
                    all_results.append( res )

        return all_results