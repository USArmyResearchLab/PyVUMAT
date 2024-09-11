# File: pyvumat/simple_ML_vumat.py
#
# Import necessary modules
import numpy as np
import torch
import configparser

class UserVumat:
    def __init__(self,config_file):

        # Create the ML model
        self.ml_model = torch.nn.Sequential(torch.nn.Linear(6,100),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(100,6))

        # Parse the INI config file and get the file path
        # to the trained weights
        parser = configparser.ConfigParser()
        parser.read(config_file)
        model_file = parser.get('Model','modelfilename')

        # Load the trained weights
        self.ml_model.load_state_dict(torch.load(model_file))

        # Set model to eval mode
        self.ml_model.eval()

    def evaluate(self, **kwargs):

        # Extract the required arguments from the keywords
        stretchNew = kwargs['stretchNew']

        # Evaluate the predicted output   
        with torch.no_grad():

            # Convert input to PyTorch tensor
            input = torch.from_numpy(stretchNew)

            # Predict stress 
            output = self.ml_model(input)

            # Update new stress. 
            stressNew = output.detach().cpu().numpy()
            
        # Only the stress is updated in this simple model so we 
        # return the old values for the other terms
        stateOld = kwargs['stateOld']
        enerInternOld = kwargs['enerInternOld']
        enerInelasOld = kwargs['enerInelasOld']        

        # Return the output arguments of the VUMAT
        return stressNew, stateOld, enerInternOld, enerInelasOld
