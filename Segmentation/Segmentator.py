import cv2

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
    def segm_Binary( cls, image, thres_val=-1 ):
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
    def process( cls, image ):
        image = cls.prepare_image( image )
        
        all_results = []
        
        ## 1 OTSU
        res = cls.segm_Otsu( image )
        all_results.append( res )

        ## 2 THRESH_BINARY
        for val in [100, 110, 120, 130, 140, 150]:
            res = cls.segm_Binary( image, thres_val=val )
            all_results.append( res )

        ## 3 ADAPTIVE_THRESHOLD
        for method in [ 'mean', 'gaussian' ]:
        
            base_blockSize = 5
            step_blockSize = 10
            range_blockSize = range(base_blockSize, base_blockSize+step_blockSize*3, step_blockSize)
            base_param2 = 3
            step_param2 = 2
            range_param2 = range(base_param2, base_param2+step_param2*3, step_param2)
            
            for param1 in range_blockSize:
                for param2 in range_param2:
                    res = cls.segm_Adaptive( image, _method=method, _block_size=param1, _subtr=param2 )
                    all_results.append( res )
                    # res = cls.segm_Binary( res, thres_val=150 )
                    # all_results.append( res )

        return all_results