import numpy as np

# add your statistics here, '_' denotes the unwrapped one
def _dice(maskgt,maskdt):
    intersect=np.logical_and(maskdt,maskgt).sum()
    dice=2*intersect/(maskdt.sum()+maskgt.sum())
    return dice

class EvalMetric(object):
    def __init__(self,masksgt,masksdt):
        self.masksgt=masksgt
        self.masksdt=masksdt
        
    def __len__(self):
        if len(self.masksgt)==len(self.masksdt):
            return len(self.masksgt)
        else:
            raise ValueError('The length of the ground truth and detections differs!')

    def wrapper(self,func):
        results=[]
        for maskgt,maskdt in zip(self.masksgt,self.masksdt):
            if isinstance(maskdt,np.ndarray) and isinstance(maskgt,np.ndarray):
                results.append(func(maskgt,maskdt))
            else:
                raise TypeError('"maskdt" or "maskgt" should be np.ndarray, but get {} and {}.' \
                                .format(type(maskdt),type(maskgt)))
                                
        return results
    
    @property
    def mean_dice(self):
        return np.mean(self.wrapper(_dice))