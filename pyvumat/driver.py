import sys
if sys.version_info[0] == 2:
    import ConfigParser as configparser
else: 
    import configparser

# Import user-defined models
from pyvumat.comm_only import CommOnlyVumat
from pyvumat.svk.svk_vumat import SvkVumat

# These modules won't work without Python 3 and PyTorch
# It's ok to skip imports if model is not used
try:
    from pyvumat.svk.svknn_vumat import SvkNnVumat
    from pyvumat.simple_ml_vumat import UserVumat
except:
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
        elif model_type == 'comm_only':
            self.model = CommOnlyVumat(config_file)
        else:
            print("\n Error: unknown model type \n")
            sys.stdout.flush()
            sys.exit()

    def evaluate(self, **kwargs):            

        # Point to the user-defined material 
        # model function
        user_function = self.model.evaluate

        return user_function(**kwargs)
        
