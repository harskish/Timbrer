### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
### Modified by Erik Härkönen and Pauli Kemppinen, 2019
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import random
import torch

class NumpyDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        self.archive_A_path = os.path.join(opt.dataroot, opt.datasets['source'])
        self.archive_B_path = os.path.join(opt.dataroot, opt.datasets['target'])

        self.archive_A = np.load(self.archive_A_path, 'r')
        self.archive_B = np.load(self.archive_B_path, 'r')
        assert self.archive_A.shape == self.archive_B.shape

        indices = list(range(self.archive_A.shape[0]))

        # Remove archive memmaps for pickling (when forking)
        self.archive_A = None
        self.archive_B = None
        
        # Test-train split
        ratio = 1/10
        random.seed(0)
        random.shuffle(indices)
        if not opt.isTrain:
            self.indices = indices[0:int(ratio*len(indices))]
            self.dataset_size = len(self.indices)
        else:
            self.indices = indices[int(ratio*len(indices)):]
            self.dataset_size = len(self.indices)
      
    def __getitem__(self, index):        
        assert index < len(self.indices), 'Trying to read out of index list'

        if self.archive_A is None:
            self.archive_A = np.load(self.archive_A_path, 'r')

        if self.archive_B is None:
            self.archive_B = np.load(self.archive_B_path, 'r')
        

        # augmentation with gaussian noise in the mel-space (to do: what is the noise in primal space?)

        A_data = self.archive_A[self.indices[index]]
        A_data = np.maximum(np.zeros_like(A_data, dtype=np.float32), A_data + np.random.randn(*A_data.shape).astype(np.float32)*.5*float(np.random.rand()))
        A_data = np.expand_dims(A_data, axis=0)

        B_data = self.archive_B[self.indices[index]]
        B_data = np.maximum(np.zeros_like(B_data, dtype=np.float32), B_data + np.random.randn(*B_data.shape).astype(np.float32)*.5*float(np.random.rand()))
        B_data = np.expand_dims(B_data, axis=0)

        A_tensor = torch.from_numpy(A_data)
        B_tensor = torch.from_numpy(B_data)

        inst_tensor = feat_tensor = 0

        res_path = 'timbrer_{}_{}'.format('train' if self.opt.isTrain else 'test', index)
        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': res_path}

        return input_dict

    def __len__(self):
        return len(self.indices) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'NumpyDataset'