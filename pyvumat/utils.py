import numpy as np
import torch

# Similar to scikit-learn's StandardScaler but performed on Torch tensors
# scale_mean and scale_std can be arrays of bools to specify scaling on each
# dimension of the data.
class TorchScaler:
    def __init__(self,scale_mean=False,scale_std=True,flatten=False,
                 dtype=torch.float32,device=None):
        self.scale_mean = scale_mean
        self.scale_std = scale_std
        self.flatten = flatten
        self.dtype=dtype
        self.device=device

        self.mean = torch.tensor(0.0,dtype=dtype).to(device)
        self.std = torch.tensor(1.0,dtype=dtype).to(device)

    def fit(self,data):

        shape = data.shape

        if self.flatten:
            data_view = data.flatten().to(self.dtype)
        else:
            data_view = data.reshape((-1,shape[-1])).to(self.dtype)

        self.mean = torch.mean(data_view,dim=0)*self.scale_mean
        self.std = torch.std(data_view,dim=0)*self.scale_std

    def set_dtype(self,dtype):
        self.dtype = dtype
        self.mean = self.mean.to(self.dtype)
        self.std = self.std.to(self.dtype)

    def transform(self, data):

        return (data - self.mean)/self.std

    def inverse_transform(self,data):

        return (data*self.std) + self.mean

    def get_state(self):

        state = {}
        state['scale_mean'] = self.scale_mean
        state['scale_std'] = self.scale_std
        state['flatten'] = self.flatten
        #prevent saving on GPU from loading on CPU
        state['mean'] = self.mean.to('cpu')
        state['std'] = self.std.to('cpu')
        state['dtype'] = self.dtype

        return state
    
    def set_state(self,state):

        self.dtype = state['dtype']
        self.scale_mean = state['scale_mean']
        self.scale_std = state['scale_std']
        self.flatten = state['flatten']
        self.mean = state['mean'].to(self.dtype).to(self.device)
        self.std = state['std'].to(self.dtype).to(self.device)

    def print_state(self):
        print("Torch Scaler State:")
        print("  Scale mean:", self.scale_mean)
        print("  Scale std:", self.scale_std)
        print("  Flatten:", self.flatten)
        print("  Mean:", self.mean.to('cpu'))
        print("  Std:", self.std.to('cpu'))

#Same as TorchScaler above, but uses Numpy instead of PyTorch        
class Scaler:
    def __init__(self,scale_mean=False,scale_std=True,flatten=False):
        self.scale_mean = scale_mean
        self.scale_std = scale_std
        self.flatten = flatten
        self.mean = 0.0
        self.std = 1.0

    def fit(self,data):
        
        shape = data.shape

        if self.flatten:
            data_view = data.flatten()
        else:
            data_view = data.reshape((-1,shape[-1]))

        self.mean = np.mean(data_view,axis=0)*self.scale_mean
    
        self.std = np.std(data_view,axis=0)*self.scale_std

        return

    def transform(self, data):

        return (data - self.mean)/self.std

    def inverse_transform(self,data):        

        return (data*self.std) + self.mean

    def get_state(self):

        state = {}
        state['scale_mean'] = self.scale_mean
        state['scale_std'] = self.scale_std
        state['flatten'] = self.flatten
        state['mean'] = self.mean
        state['std'] = self.std

        return state

    def set_state(self,state):

        self.scale_mean = state['scale_mean']
        self.scale_std = state['scale_std']
        self.flatten = state['flatten']
        self.mean = state['mean']
        self.std = state['std']

        return

    def print_state(self):
        print("Scaler State:")
        print("  Scale mean:", self.scale_mean)        
        print("  Scale std:", self.scale_std)
        print("  Flatten:", self.flatten)        
        print("  Mean:", self.mean)        
        print("  Std:", self.std)
        return

