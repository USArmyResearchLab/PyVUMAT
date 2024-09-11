# Python script to generate training data for the svknn_vumat.py.
# Generates random stretch tensors and Poisson's ratios then computes
# corresponding stresses based on the Saint Venant-Kirchhoff hyperelastic
# material model

import numpy as np
from pyvumat.svk.svk_vumat import SvkVumat

# Define Input Parameters (modified by user)
num_points = 20000
strain_low, strain_high = (-0.01, 0.01)
poisson_low, poisson_high = (0.05, 0.45)
out_file = 'Data_SVK_strain1_20K.csv'

# Generate training data
# U = [Uxx, Uyy, Uzz, Uxy, Uyz, Uxz]
stretch = np.random.rand(num_points,6)
stretch = strain_low + (strain_high-strain_low)*stretch
stretch[:,:3] += 1.0
poisson = np.random.rand(num_points,1)
poisson = poisson_low + (poisson_high-poisson_low)*poisson

# Instantiate the model 
config_file = None
model = SvkVumat(config_file)

# Evaluate stress at strains
stateOld = np.zeros((num_points,2))
stress = np.zeros((num_points,6))

# Since the Poisson's ratio (and therefore props array) is
# different for each point, we can not evaluate the stresses
# in one batch. We need to evaluate them separately, passing in
# the corresponding Poisson's ratio.
youngsMod = 1.0
for i in range(num_points):
    props = np.array([youngsMod, poisson[i,0]])
    stretchNew = stretch[i].reshape(1,6)
    kwargs = {'stretchNew':stretchNew, 
              'props':props, 
              'stateOld':stateOld}

    stressNew, _, _, _ = model.evaluate(**kwargs)

    stress[i,:] = stressNew.reshape(6)

# Write the input and output data to a file
index = np.arange(num_points).reshape(-1,1)
data = np.hstack((index, poisson, stretch, stress))
header = ("Index, Poisson's Ratio, Uxx, Uyy, Uzz, Uxy, Uyz, Uxz, " +
          "Sxx, Syy, Szz, Sxy, Syz, Sxz")
np.savetxt(out_file,data,delimiter=', ',header=header)
