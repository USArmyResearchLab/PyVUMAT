import numpy as np
import argparse
import torch
from  torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from pyvumat.svk.fcnn import FCNN_Driver

# Set random seed so training is reproducible
torch.manual_seed(42)

# Construct the argument parse and parse the command line arguments
ap = argparse.ArgumentParser(description='Train a simple fully ' + 
                             'connected NN with PyTorch')

ap.add_argument("--layers-sizes", type=int, required=True, nargs='+', 
                help="List of number of nodes for each layer")
ap.add_argument("-i","--in-file", type=str, required=True,
                help="Name of input file")
ap.add_argument("-o","--out-file", type=str, required=True,
                help="Prefix of output files")
ap.add_argument("--epochs", default=1000, type=int, 
                help="Number of epochs [1000]")
ap.add_argument("--batch", default=100, type=int, 
                help="Batch size [100]")
ap.add_argument("--max-lr", default=0.001, type=float, 
                help="Max learning rate [0.001]")
ap.add_argument("--min-lr", default=1.0e-6, type=float, 
                help="Min learning rate [1.0e-6]")
ap.add_argument("--train-ratio", default=0.9, type=float, 
                help="Ratio of data to use for training [0.9]")
ap.add_argument("--test-ratio", default=0.1, type=float, 
                help="Ratio of data to use for testing [0.1]")
args = ap.parse_args()

# Build the model using the command line arguments
model = FCNN_Driver(cmd_args=args)
    
# Read the files containing training and test data
raw_data = np.loadtxt(args.in_file, 
                      skiprows=1, delimiter=',')

# Ignore the index column
data = raw_data[:,1:]

# Make sure the length of the data is equal to the 
# input + output dimensions
input_dim = args.layers_sizes[0]
output_dim = args.layers_sizes[-1]

num_points, dim = data.shape
if not dim == (input_dim + output_dim):
    print("Error: size of training data is", dim, 
          ". Expected ", input_dim+output_dim)

# Split the data into inputs and outputs
data_input = data[:,:input_dim]
data_output = data[:,input_dim:(input_dim+output_dim)]
    
# Scale input and output data and convert to torch tensors
data_input = model.process_input(data_input, fit_scaler=True)
data_output = model.process_output(data_output, fit_scaler=True)

# Separate the training and test data
train_size = int(num_points*args.train_ratio)
test_size = int(num_points*args.test_ratio)
test_start = num_points-test_size

x_train = data_input[0:train_size]
y_train = data_output[0:train_size]

x_test = data_input[test_start:]
y_test  = data_output[test_start:]

# Wrap traning data in loader
train_loader = DataLoader(TensorDataset(x_train, y_train), 
                          batch_size=args.batch, 
                          shuffle=True)

# Optimizer and learning rate scheduler
optimizer = torch.optim.Adam(model.nn.parameters(), 
                             lr=args.max_lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                 args.epochs, 
                                                 args.min_lr)

# Define loss function
loss_func = nn.MSELoss()

error_file = open(args.out_file + "_error.csv","w")
error_file.write(str(args)+"\n\n\n")
error_file.write("Epoch, Training Error, Test Error, \n")

train_err = np.zeros((args.epochs,))
test_err = np.zeros((args.epochs,))
for ep in range(args.epochs):
    train_loss, test_loss = 0.0, 0.0
    for x, y in train_loader:
        optimizer.zero_grad()
        y_approx = model.nn(x)        
        loss = loss_func(y_approx,y)
        loss.backward()                                
        train_loss = train_loss + loss.item()
        optimizer.step()                        
    scheduler.step()
        
    with torch.no_grad():
        y_test_approx = model.nn(x_test)
        t_loss = loss_func(y_test_approx,y_test)
        test_loss = t_loss.item()

    train_err[ep] = train_loss/len(train_loader)
    test_err[ep]  = test_loss
    
    print(ep, train_err[ep],test_err[ep])
    error_file.write("{}, {}, {}\n".format(ep,train_err[ep],
                                           test_err[ep]))
    error_file.flush()
error_file.close()

# Save the model to a file
model.save(args.out_file+"_model.pth")
