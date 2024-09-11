# Defining the architecture for the fully connected (FC) neural net (NN)
import sys
import numpy as np
import torch
import torch.nn as nn
from pyvumat.utils import TorchScaler

# Define the fully connected neural network (FCNN) model
class FCNN(nn.Module):
    def __init__(self, layers_sizes):
        super(FCNN, self).__init__()

        # construct the layers for the feed-forward NN
        layers = [nn.Linear(layers_sizes[0],
                            layers_sizes[1])]
        for j in range(1,len(layers_sizes) - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(layers_sizes[j], 
                                    layers_sizes[j+1]))

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
        print("DEVICE:", self.device, flush=True)

        # Initialize the scalers
        self.output_scaler = TorchScaler(scale_mean=True,scale_std=True,
                                         device=self.device)
        self.input_scaler = TorchScaler(scale_mean=True,scale_std=True,
                                        device=self.device)

        # Get arguments for FCNN model from the command line (cmd_args) or
        # saved from previously trained model (saved_file)
        if cmd_args is not None:
            self.layers_sizes = cmd_args.layers_sizes

            # Create the NN 
            self.nn = FCNN(self.layers_sizes)

            # Set the type
            if cmd_args.t == 32:
                self.dtype = torch.float32
            elif cmd_args.t == 64:
                self.dtype = torch.float64
            else:
                print("Error: Unsupported number of bits for "
                      "pytorch type",cmd_args.t,"Using float32",
                      flush=True)
                self.dtype = torch.float32

        elif saved_file is not None:
            saved_model = torch.load(saved_file,
                                     map_location=self.device)

            # Load the size of each NN layer
            self.layers_sizes = saved_model['layers_sizes']

            # Create the NN 
            self.nn = FCNN(self.layers_sizes)

            # Set the type
            self.dtype = saved_model['dtype']

            # Load weights & biases
            self.nn.load_state_dict(saved_model['model_state'])

            # Load the parameters for scaling the inputs and outputs 
            # that were fit to the training data
            self.input_scaler.set_state(saved_model['input_scale'])
            self.output_scaler.set_state(saved_model['output_scale'])

        else:
            print("Error: Constructor for FCNN_Driver requires either",
                  "cmd_args or saved_file argument")
            sys.exit(1)
        
        # Convert model to appropriate type
        self.input_scaler.set_dtype(self.dtype)
        self.output_scaler.set_dtype(self.dtype)
        self.nn.to(self.dtype).to(self.device)

    # Save the model
    def save(self,out_file_name):
        model_data = {
            "layers_sizes": self.layers_sizes,
            "input_scale": self.input_scaler.get_state(),
            "output_scale": self.output_scaler.get_state(),
            "model_state": self.nn.state_dict(),
            "dtype": self.dtype,
        }
        torch.save(model_data,out_file_name)
        return

    # Convert tensor to appropriate type and move to appropriate device
    def to(self,tensor):
        return tensor.to(self.dtype).to(self.device)

    def process_data(self,data,scaler,expected_dim,fit_scaler=False):
        batch_size, dim = data.shape        
        if not dim == expected_dim:
            print("Error: inconsistent input dimension when preprocessing")
            print(dim, expected_dim)
            sys.exit(1)

        # Convert data to torch tensor
        if torch.is_tensor(data):
            pyt_data = self.to(data)
        else:
            pyt_data = self.to(torch.from_numpy(data))
        
        # Fit the scaler if requested
        if fit_scaler:
            scaler.fit(pyt_data)

        # Scale the data
        return_data = scaler.transform(pyt_data)

        # Convert to torch tensor
        return self.to(return_data.view(batch_size,
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

        
