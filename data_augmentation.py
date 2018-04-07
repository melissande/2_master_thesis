import numpy as np
from imgaug import augmenters as iaa
from numpy import newaxis
class Flip(object):
    """Flip ratio of the image Left/Right and up/down

    Args:
        ratio (float): how much of the image is flipped
    """

    def __init__(self,ratio):

        self.ratio = ratio

    def __call__(self, sample):
        X, Y = sample['input'], sample['groundtruth']
        seq = iaa.Sequential([iaa.Fliplr(self.ratio),iaa.Flipud(self.ratio)])
        data_tot=np.concatenate((X.astype('uint8'),Y.astype('uint8')),axis=2)
        data_tot=seq.augment_images(data_tot[newaxis,:,:,:])
        data_tot=np.squeeze(data_tot)
        X=data_tot[:,:,:X.shape[2]]
        Y=data_tot[:,:,X.shape[2]:]
        return {'input': X, 'groundtruth': Y}

class Rotate(object):
    """Rotate of random angle between 0 and max_rot of the image

    Args:
        max_rot (int): how much of the image is rotated
    """

    def __init__(self,max_rot):

        self.max_rot = max_rot

    def __call__(self, sample):
        X, Y = sample['input'], sample['groundtruth']
        seq = iaa.Sequential(iaa.Affine(rotate=(0, max_rot))
        data_tot=np.concatenate((X.astype('uint8'),Y.astype('uint8')),axis=2)
        data_tot=seq.augment_images(data_tot[newaxis,:,:,:])
        data_tot=np.squeeze(data_tot)
        X=data_tot[:,:,:X.shape[2]]
        Y=data_tot[:,:,X.shape[2]:]
        return {'input': X, 'groundtruth': Y}
    
    
    
class Rescale(object):
    """Crop and Pad = rescale effect
    """

    def __init__(self,ratio):

        self.ratio = ratio

    def __call__(self, sample):
        X, Y = sample['input'], sample['groundtruth']
        seq = iaa.Sequential(iaa.CropAndPad(percent=(-0.25, 0.25))
        data_tot=np.concatenate((X.astype('uint8'),Y.astype('uint8')),axis=2)
        data_tot=seq.augment_images(data_tot[newaxis,:,:,:])
        data_tot=np.squeeze(data_tot)
        X=data_tot[:,:,:X.shape[2]]
        Y=data_tot[:,:,X.shape[2]:]
        return {'input': X, 'groundtruth': Y}
    