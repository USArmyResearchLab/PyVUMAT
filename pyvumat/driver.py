import sys
import os

if sys.version_info[0] == 2:
    import ConfigParser as configparser
else: 
    import configparser

import numpy as np
    
# Import user-defined models
from pyvumat.svk.svk_vumat import SvkVumat
from pyvumat.identity_vumat import IdentityVumat

# These modules won't work without Python 3 and PyTorch
# It's ok to skip imports if model is not used
try:
    from pyvumat.svk.svknn_vumat import SvkNnVumat
    from pyvumat.simple_ml_vumat import UserVumat
except:
    print("\nWARNING: Could not load PyTorch VUMATs.")
    print('If you plan to use these models, move import',
          'statements outside of "try" \nstatement',
          'to determine issue.\n')
    sys.stdout.flush()
    pass

class Driver:

    def __init__(self,config_file=None):

        if config_file == None:
            print("\nError: This implementation of the Driver",
                  "class requires an INI file\n",
                  "      specified by the environment variable",
                  "PYVUMAT_CONF_FILE\n")
            sys.stdout.flush()
            sys.exit()

        # Check if the configuration file exists
        if not os.path.isfile(config_file):
            print("\nError: This implementation of the Driver",
                  "class requires an INI file.\n",
                  config_file, "does not exit.")
            sys.stdout.flush()
            sys.exit()

            
        # Parse the options from the ini file
        parser = configparser.ConfigParser()
        parser.read(config_file)

        # Choose the model
        model_type = parser.get('Driver','model')
        if model_type == 'simple_ml':
            self.model = UserVumat(config_file)
        elif model_type == 'svk':
            self.model = SvkVumat(config_file)
        elif model_type == 'svknn':
            self.model = SvkNnVumat(config_file)
        elif model_type == 'identity':
            self.model = IdentityVumat(config_file)
        else:
            print("\n Error: unknown model type \n")
            sys.stdout.flush()
            sys.exit()    

        # Get verbosity flag if provided in INI file
        if parser.has_option('Driver','Verbose'):
            self.verbose = parser.getint('Driver','verbose')
        else:
            self.verbose = 0

    def evaluate(self, **kwargs):            

        if self.verbose > 0:
            print("Input:")
            for key, value in kwargs.items():
                if np.ndim(value) < 1:
                    print(key,value)
                else:
                    print(key,value.shape)            
            if self.verbose > 1:
                print(kwargs)                
            sys.stdout.flush()

        # Point to the user-defined material 
        # model function
        user_function = self.model.evaluate

        stress, state, enerIntern, enerInelas = user_function(**kwargs)

        # Set Fortran ordering for 2D arrays
        stress = np.asfortranarray(stress)
        state = np.asfortranarray(state)
            
        if self.verbose > 0:
            print("Output:")
            print("Stress:", stress.shape)
            print("State:", state.shape)
            print("EnerIntern:", enerIntern.shape)
            print("EnerInelas:", enerInelas.shape)

            if self.verbose > 1:
                print("Stress:", stress)
                print("State:", state)
                print("EnerIntern:", enerIntern)
                print("EnerInelas:", enerInelas)
            sys.stdout.flush()                           

        return stress, state, enerIntern, enerInelas
