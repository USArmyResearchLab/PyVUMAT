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
        num_points = stretchNew.shape[1]
        poisson = np.full((1,num_points),props[1])
        input = np.vstack((poisson,
                           stretchNew))

        #Evaluate the predicted output        
        with torch.no_grad():
            # Convert the input to a PyTorch tensor and perform 
            # scaling consistent with scaling of training data.
            # Inputs have dimension [n, num_points], transpose to 
            # [num_points, n] as required in pytorch.
            pyt_input = self.model.process_input(input.T)
            
            # Predict the stress from the ML model
            stress_out = self.model.nn(pyt_input)

            # Apply inverse scaling on stress
            stress_out = self.model.inverse_scale_output(stress_out)

        # Output is [num_points, n]. Transpose to [n, num_points] and 
        # copy to ensure contiguous memory layout for return array
        stressNew = np.transpose(stress_out).copy()
        
        # Scale by E since training was done with E=1
        stressNew *= youngsMod

        # Only the stress is updated in this model so we return the 
        # old values for the other terms
        stateOld = kwargs['stateOld']
        enerInternOld = kwargs['enerInternOld']
        enerInelasOld = kwargs['enerInelasOld']        

        return stressNew, stateOld, enerInternOld, enerInelasOld
