from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import h5py
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
plt.ion()   # interactive mode

PATH_INPUT='INPUT/'
PATH_OUTPUT='OUTPUT/'
NB_CLASSES=2

def _parse_image(path_input,path_output,nb_classes):
    '''
    Reads and saves as as an array image input and output
    :paths_input path of the input image that have to be read  
    :paths_output path of the output image that have to be read  
    returns input and output image as array
    '''
    
    with h5py.File(path_input, 'r') as hf:
            X =np.array(hf.get('data'))
    with h5py.File(path_output, 'r') as hf:
            Y_build=np.array(hf.get('data'))
            Y_build=(Y_build>0).astype(int)
            Y_other= (1-Y_build).astype(int)
            Y=np.stack((Y_other,Y_build),axis=2)
            
    return X,Y


class Dataset_sat(Dataset):
    """Satellite images dataset with rastered footprints in groundtruth."""

    def __init__(self,paths_input: np.ndarray,paths_output: np.ndarray,nb_classes: int,transform=None):
        """
        Args:
            paths_input: paths of the patch images  in input
            paths_output: paths of the patch groundtruth  in input
            nb_classes: number of classes in the rasterized version of groundtruth
            transform: for data augmentation
            
        """
        self.paths_input = paths_input
        self.paths_output = paths_output
        self.nb_classes=nb_classes
        self.transform = transform
#         self.file_check=open('check.txt','w')

    @classmethod
    def from_root_folder(cls, root_folder: str, nb_classes: int,*,transform=None, max_data_size:  int = None):
        paths_input = []
        paths_output=[]
        
        
        for filename in sorted(os.listdir(root_folder+PATH_INPUT))[:max_data_size]:
            paths_input.append(os.path.join(root_folder+PATH_INPUT, filename))

        for filename in sorted(os.listdir(root_folder+PATH_OUTPUT))[:max_data_size]:

            paths_output.append(os.path.join(root_folder+PATH_OUTPUT, filename))
        

        return Dataset_sat(np.asarray(paths_input), np.asarray(paths_output),nb_classes,transform)

    def __len__(self):
        return len(self.paths_input)
    


    def __getitem__(self, idx):
        
        X,Y=_parse_image(self.paths_input[idx],self.paths_output[idx],self.nb_classes)
        sample = {'input': X, 'groundtruth': Y}

#         self.file_check.write(str(self.paths_input[idx].split('input_')[-1:]))
 
        if self.transform:
            sample = self.transform(sample)
  

        return sample


