# Defining the architecture for the fully connected (FC) neural net (NN)
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Define the fully connected neural network (FCNN) model
class FCNN(nn.Module):
    def __init__(self, layers_sizes):
        super(FCNN, self).__init__()

        # construct the layers for the feed-forward NN
        layers = []
        for j in range(len(layers_sizes) - 1):
            layers.append(nn.Linear(layers_sizes[j], 
                                    layers_sizes[j+1]))
            layers.append(nn.ReLU())
        
        # Remove activation from the output layer
        layers.pop()

        self.layers = nn.Sequential(*layers)

    def forward(self, input):

        return self.layers(input)

# Create the class to :
#
#  1) Construct the FCNN model (either from scratch through the command-line 
#     arguments or loaded from a saved pytorch file)
#  2) Save the model and relevant parameters to a pytorch file so it can 
#     be reloaded later
#  3) Scale and inverse scale the inputs and outputs
class FCNN_Driver():
    def __init__(self, cmd_args=None,saved_file=None):
        
        # Check for gpu
        self.device = torch.device('cuda:0' if torch.cuda.is_available() 
                                   else 'cpu')

        # Initialize the scalers
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

        # Set the type to float32
        self.type = torch.float32

        # Get arguments for FCNN model from the command line (cmd_args) or
        # saved from previously trained model (saved_file)
        if cmd_args is not None:
            self.layers_sizes = cmd_args.layers_sizes

            # Create the NN 
            self.nn = FCNN(self.layers_sizes)

        elif saved_file is not None:
            saved_model = torch.load(saved_file,
                                     map_location=self.device)

            # Load the size of each NN layer
            self.layers_sizes = saved_model['layers_sizes']

            # Create the NN 
            self.nn = FCNN(self.layers_sizes)

            # Load weights & biases
            self.nn.load_state_dict(saved_model['model_state'])

            # Load the parameters for scaling the inputs and outputs 
            # that were fit to the training data
            self.input_scaler.__setstate__(saved_model['input_scale'])
            self.output_scaler.__setstate__(saved_model['output_scale'])

        else:
            print("Error: Constructor for FCNN_Driver requires either",
                  "cmd_args or saved_file argument")
            sys.exit(1)
        
        # Convert model to appropriate type
        self.nn.to(self.type).to(self.device)

    # Save the model
    def save(self,out_file_name):
        model_data = {
            "layers_sizes": self.layers_sizes,
            "input_scale": self.input_scaler.__getstate__(),
            "output_scale": self.output_scaler.__getstate__(),
            "model_state": self.nn.state_dict()
        }
        torch.save(model_data,out_file_name)
        return

    # Convert tensor to appropriate type and move to appropriate device
    def to(self,tensor):
        return tensor.to(self.type).to(self.device)

    def process_data(self,data,scaler,expected_dim,fit_scaler=False):
        batch_size, dim = data.shape        
        if not dim == expected_dim:
            print("Error: inconsistent input dimension when preprocessing")
            print(dim, expected_dim)
            sys.exit(1)
                
        # Fit the scaler if requested
        if fit_scaler:
            scaler.fit(data)

        # Scale the data
        return_data = scaler.transform(data)

        # Convert to torch tensor
        return self.to(torch.from_numpy(return_data).view(batch_size,
                                                                 dim))        
        
    # Preprocess input data 
    def process_input(self,input_data,fit_scaler=False):
        return self.process_data(input_data, self.input_scaler, 
                                        self.layers_sizes[0],fit_scaler)
                            
    # Preprocess output data 
    def process_output(self,output_data,fit_scaler=False):
        return self.process_data(output_data, self.output_scaler, 
                                        self.layers_sizes[-1],fit_scaler)

    # Perform inverse of scaling the input data
    def inverse_scale_input(self,scaled_input):
        return self.input_scaler.inverse_transform(scaled_input)

    # Perform inverse of scaling the output data
    def inverse_scale_output(self,scaled_output):
        return self.output_scaler.inverse_transform(scaled_output)

        
