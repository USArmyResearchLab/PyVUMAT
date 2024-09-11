import numpy as np

class IdentityVumat:
    def __init__(self,config_file=None):
        # Nothing required of the constructor
        pass
    
    def evaluate(self, **kwargs):
        
        # Multiply U-I by identity to get stress
        stressNew = np.copy(kwargs['stretchNew'])
        stressNew[:,0:3] -= 1.0
        
        # Extract the required arguments from the keywords
        stateOld = kwargs['stateOld']
        enerInternOld = kwargs['enerInternOld']
        enerInelasOld = kwargs['enerInelasOld']
        
        # Return input values as the output values
        return stressNew, stateOld, enerInternOld, enerInelasOld
