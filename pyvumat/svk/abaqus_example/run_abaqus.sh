#!/bin/bash

# 1) Add the PyVUMAT base directory to PYTHONPATH so that Python can find the
#    driver and specific PyVUMAT model. Note that the user will also need to
#    add the location of their user-defined model if they are not using one of
#    the provided models (i.e. the analytical SVK or ML SVK) or if they have
#    not placed their model within <PyVUMAT-base-dir>.
#
#    <PyVUMAT-base-dir> should contain pyvumat/driver.py
VUMAT_DIR=<PyVUMAT-base-dir>
export PYTHONPATH=${VUMAT_DIR}:${PYTHONPATH}

# 2) If using a PyVUMAT model that relies on Python packages beyond NumPy
#    (e.g. PyTorch, scikit-learn) it is easiest to create a separate environment
#    with the additional packages and add the necessary paths to PYTHONPATH.
#    Note that when installing Python packages to use with Abaqus, you should
#    use a Python version as close as possible to the version provided in your
#    Abaqus install.
#
#    This example assumes a conda environment called 'pytorch3.10' that uses
#    Python 3.10 and is installed in ~/.conda/envs

conda activate pytorch3.10
PYTHON_ENV_DIR=${HOME}/.conda/envs/pytorch3.10/
PYTHON_VERSION=3.10
export PYTHONPATH=${PYTHON_ENV_DIR}/lib/python${PYTHON_VERSION}/site-packages:${PYTHONPATH}

# 3) Set the environment variable with the path to your PyVUMAT INI
#    configuration file.
#
#    This example uses the file provided in the source for running the ML SVK
#    model. The user must modify the 'ModelFileName' keyword with the path to
#    the saved PyTorch model.

export PYVUMAT_CONF_FILE=${VUMAT_DIR}/pyvumat/svk/abaqus_example/pyVumat_Conf_svkNN.ini

# 4) Make sure you have a local 'abaqus_v6.env' file that adds the header and
#    library paths for Abaqus' Python install as described in the User's Guide.

# 5) Run Abaqus
abaqus job=example user=${VUMAT_DIR}/pyVUMAT.cpp verbose=2 double=both
