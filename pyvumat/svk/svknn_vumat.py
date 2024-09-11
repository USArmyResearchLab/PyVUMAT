import sys
import os

# Depending on how the vumat library is loaded by the FEA code,
# may need to change the dlopen flags. Not necessary for Abaqus.
#sys.setdlopenflags(os.RTLD_NOW | os.RTLD_DEEPBIND)

import numpy as np
import torch
import configparser
from pyvumat.svk.fcnn import FCNN_Driver

class SvkNnVumat:
    def __init__(self,config_file):

        # Read the configuration file
        parser = configparser.ConfigParser()
        parser.read(config_file)
        model_file = parser.get('Model','modelfilename')
        self.verbose = parser.getint('Model','verbose',
                                     fallback=0)

        # If the number of threads is specified, overwrite
        # default value
        threads = parser.getint('Model','threads',
                                fallback=torch.get_num_threads())
        torch.set_num_threads(threads)

        # Option to turn on TensorFloat32 math mode if using
        # NVIDIA's Ampere GPUs or newer. Trades accuracy for
        # improved performance.
        #torch.backends.cuda.matmul.allow_tf32 = True
        
        # Create the model, load the trained weights and 
        # scaling parameters
        self.model = FCNN_Driver(saved_file=model_file)

        # Set model to eval mode
        self.model.nn.eval()

        if self.verbose > 0:
            print(self.model.nn, flush=True)

    def evaluate(self, **kwargs):

        # Extract the required arguments from the keywords
        stretchNew = kwargs['stretchNew']
        props = kwargs['props']
        youngsMod = props[0]

        # Add Poisson's ratio as an input to the model   
        num_points = stretchNew.shape[0]
        poisson = np.full((num_points,1),props[1])
        input = np.hstack((poisson,
                           stretchNew))

        #Evaluate the predicted output        
        with torch.no_grad():
            # Convert the input to a PyTorch tensor and perform 
            # scaling consistent with scaling of training data.
            pyt_input = self.model.process_input(input)
            
            # Predict the stress from the ML model
            stress_out = self.model.nn(pyt_input)

            # Apply inverse scaling on stress
            stress_out = self.model.inverse_scale_output(stress_out)

            # Convert to NumPy double array
            stressNew = stress_out.cpu().numpy().astype('float64')
            
        # Scale by E since training was done with E=1
        stressNew *= youngsMod

        # Only the stress is updated in this model so we return the 
        # old values for the other terms
        stateOld = kwargs['stateOld']
        enerInternOld = kwargs['enerInternOld']
        enerInelasOld = kwargs['enerInelasOld']        

        return stressNew, stateOld, enerInternOld, enerInelasOld
