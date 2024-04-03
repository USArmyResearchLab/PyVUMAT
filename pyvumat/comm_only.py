import sys
import numpy as np

VERSION = sys.version_info[0]
if VERSION == 2:
    import ConfigParser as configparser
elif VERSION == 3:
    import configparser

class CommOnlyVumat:
    def __init__(self,config_file):
        self.config_file = config_file

        # Read INI file
        parser = configparser.ConfigParser({'printargs':0})
        parser.read(self.config_file)

        # Get verbosity flag
        self.verbose = 0
        if parser.has_option('Model','verbose'):
            self.verbose = parser.getint('Model','verbose')

        if self.verbose > 0:
            print("Constructed CommOnlyVumat")
            sys.stdout.flush()

    def evaluate(self, **kwargs):
        
        # Print the input arguments coming from keywords
        if self.verbose > 0:
            for key, value in kwargs.items():
                if np.ndim(value) < 1:
                    print(key,value)
                else:
                    print(key,value.shape)
            sys.stdout.flush()

        # Extract the required arguments from the keywords
        stressOld = kwargs['stressOld']
        stateOld = kwargs['stateOld']
        enerInternOld = kwargs['enerInternOld']
        enerInelasOld = kwargs['enerInelasOld']
        
        # Return input values as the output values
        return stressOld, stateOld, enerInternOld, enerInelasOld
