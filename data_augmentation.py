import numpy as np
from imgaug import augmenters as iaa
from numpy import newaxis
import torch
class Flip(object):
    """Flip ratio of the image Left/Right and up/down

    Args:
        ratio (float): how much of the image is flipped
    """

    def __init__(self,ratio):

        self.ratio = ratio

    def __call__(self, sample):
        X, Y = sample['input'], sample['groundtruth']
        Y=np.argmax(Y, 2)[:,:,newaxis]
        X,Y=(X-np.amin(X))/(np.amax(X)-np.amin(X))*255,(Y-np.amin(Y))/(np.amax(Y)-np.amin(Y))*255
        seq = iaa.OneOf([iaa.Fliplr(self.ratio),iaa.Flipud(self.ratio)])
        data_tot=np.concatenate((X.astype('uint8'),Y.astype('uint8')),axis=2)
        data_tot=seq.augment_images(data_tot[newaxis,:,:,:])
        data_tot=np.squeeze(data_tot)
        X=data_tot[:,:,:X.shape[2]]
        Y=data_tot[:,:,-1]
        X,Y=(X-np.amin(X))/(np.amax(X)-np.amin(X)),(Y-np.amin(Y))/(np.amax(Y)-np.amin(Y))
        Y_build=(Y>0).astype(int)
        Y_other= (1-Y_build).astype(int)
        Y=np.stack((Y_other,Y_build),axis=2)
        return {'input': X, 'groundtruth': Y}

class Rotate(object):
    """Rotate of random angle between 0 and max_rot of the image

    Args:
        max_rot (int): how much of the image is rotated (+max_rot and -max_rot)
    """

    def __init__(self,max_rot):

        self.max_rot = max_rot

    def __call__(self, sample):
        X, Y = sample['input'], sample['groundtruth']
        Y=np.argmax(Y, 2)[:,:,newaxis]
        X,Y=(X-np.amin(X))/(np.amax(X)-np.amin(X))*255,(Y-np.amin(Y))/(np.amax(Y)-np.amin(Y))*255
        seq = iaa.OneOf(iaa.Affine(rotate=self.max_rot),iaa.Affine(rotate=-self.max_rot))
        data_tot=np.concatenate((X.astype('uint8'),Y.astype('uint8')),axis=2)
        data_tot=seq.augment_images(data_tot[newaxis,:,:,:])
        data_tot=np.squeeze(data_tot)
        X=data_tot[:,:,:X.shape[2]]
        Y=data_tot[:,:,-1]
        X,Y=(X-np.amin(X))/(np.amax(X)-np.amin(X)),(Y-np.amin(Y))/(np.amax(Y)-np.amin(Y))
        Y_build=(Y>0).astype(int)
        Y_other= (1-Y_build).astype(int)
        Y=np.stack((Y_other,Y_build),axis=2)
        return {'input': X, 'groundtruth': Y}
    
    
    
class Rescale(object):
    """Crop and Pad = rescale effect
     Args:
        ratio (float): how much of the image is rescaled
    """

    def __init__(self,ratio):
        self.ratio=ratio
        
    def __call__(self, sample):
        X, Y = sample['input'], sample['groundtruth']
        Y=np.argmax(Y, 2)[:,:,newaxis]
        X,Y=(X-np.amin(X))/(np.amax(X)-np.amin(X))*255,(Y-np.amin(Y))/(np.amax(Y)-np.amin(Y))*255
        seq = iaa.CropAndPad(percent=(-self.ratio, self.ratio))
        data_tot=np.concatenate((X.astype('uint8'),Y.astype('uint8')),axis=2)
        data_tot=seq.augment_images(data_tot[newaxis,:,:,:])
        data_tot=np.squeeze(data_tot)
        X=data_tot[:,:,:X.shape[2]]
        Y=data_tot[:,:,-1]
        X,Y=(X-np.amin(X))/(np.amax(X)-np.amin(X)),(Y-np.amin(Y))/(np.amax(Y)-np.amin(Y))
        Y_build=(Y>0).astype(int)
        Y_other= (1-Y_build).astype(int)
        Y=np.stack((Y_other,Y_build),axis=2)
        return {'input': X, 'groundtruth': Y}
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        X, Y = sample['input'], sample['groundtruth']
        return {'input': torch.from_numpy(X),
                'groundtruth': torch.from_numpy(Y)}