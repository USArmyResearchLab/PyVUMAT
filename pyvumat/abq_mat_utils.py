# Collection of utility functions for vectorized operations on 
# flattened matricies with Abaqus ordering. 
#
# Entry A[i,j] of the flattened matrices corresponds to the j^th
# component of the i^th matrix (i.e. element/block)
#
# Currently limited to 3D symmetric matrices of size Nx6 or
# 3D nonsymmetric matricies of size Nx9

import numpy as np

def get_indices(A):
    """
    Get index mapping for symmetric and nonsymmetric 
    representations.
    """

    indices = np.arange(9)

    if A.shape[1] == 6:
        indices[6:] = np.arange(3,6)

    # Fixme: Add some error checking to make sure size is 6 or 9

    return indices

def abq_det(A):
    """
    Compute the determinant of all matricies in A

    Returns 1D array of determinant values
    """

    A_index = get_indices(A)

    det = (A[:,0]*(A[:,1]*A[:,2] - A[:,4]*A[:,A_index[7]])
           - A[:,3]*(A[:,A_index[6]]*A[:,2] - A[:,4]*A[:,5])
           + A[:,A_index[8]]*(A[:,A_index[6]]*A[:,A_index[7]] - A[:,1]*A[:,5]))

    return det

def abq_mat_mult(A, B):
    """
    Perform multiplication of flattened matrix 
    representations A and B.

    A can be symmetric (Nx6) or nonsymmetric (Nx9)
    B can be symmetric (Nx6) or nonsymmetric (Nx9)

    Returns a nonsymmetric (Nx9) flattened respresentation
    of matrix multiplications.
    """

    A_index = get_indices(A)
    B_index = get_indices(B)

    C = np.zeros((A.shape[0],9))

    C[:,0] = A[:,0]*B[:,0] + A[:,3]*B[:,B_index[6]] + A[:,A_index[8]]*B[:,5]
    C[:,1] = A[:,A_index[6]]*B[:,3] + A[:,1]*B[:,1] + A[:,4]*B[:,B_index[7]]
    C[:,2] = A[:,5]*B[:,B_index[8]] + A[:,A_index[7]]*B[:,4] + A[:,2]*B[:,2]
    C[:,3] = A[:,0]*B[:,3] + A[:,3]*B[:,1] + A[:,A_index[8]]*B[:,B_index[7]]
    C[:,4] = A[:,A_index[6]]*B[:,B_index[8]] + A[:,1]*B[:,4] + A[:,4]*B[:,2]
    C[:,5] = A[:,5]*B[:,0] + A[:,A_index[7]]*B[:,B_index[6]] + A[:,2]*B[:,5]
    C[:,6] = A[:,A_index[6]]*B[:,0] + A[:,1]*B[:,B_index[6]] + A[:,4]*B[:,5]
    C[:,7] = A[:,5]*B[:,3] + A[:,A_index[7]]*B[:,1] + A[:,2]*B[:,B_index[7]]
    C[:,8] = A[:,0]*B[:,B_index[8]] + A[:,3]*B[:,4] + A[:,A_index[8]]*B[:,2]

    return C

def abq_vec_to_mat(vec):
    """
    Converts a 2D flattened representation of matrices to a 3D
    tensor representation.

    The input vector (vec) can be symmetric (Nx6) or 
    nonsymmetric (Nx9)

    Returns 3D array of size Nx3x3
    """

    indices = get_indices(vec)
    num_points = vec.shape[0]
    mat = np.zeros((num_points,3,3))

    mat[:,0,0] = vec[:,0]
    mat[:,1,1] = vec[:,1]
    mat[:,2,2] = vec[:,2]
    mat[:,0,1] = vec[:,3]
    mat[:,1,2] = vec[:,4]
    mat[:,2,0] = vec[:,5]
    mat[:,1,0] = vec[:,indices[6]]
    mat[:,2,1] = vec[:,indices[7]]
    mat[:,0,2] = vec[:,indices[8]]

    return mat

def abq_mat_to_full_vec(mat):
    """
    Converts a 3D nonsymmetric tensor representation of 
    matrices to flattened representation.

    The input matrices (mat) are of size Nx3x3

    Returns 2D array of flattened nonsymmetric representation 
    of size Nx9
    """

    num_points = mat.shape[0]
    vec = np.zeros((num_points,9))

    vec[:,0] = mat[:,0,0]
    vec[:,1] = mat[:,1,1]
    vec[:,2] = mat[:,2,2]
    vec[:,3] = mat[:,0,1]
    vec[:,4] = mat[:,1,2]
    vec[:,5] = mat[:,2,0]
    vec[:,6] = mat[:,1,0]
    vec[:,7] = mat[:,2,1]
    vec[:,8] = mat[:,0,2]
    
    return vec

def abq_mat_to_symm_vec(mat):
    """
    Converts a 3D symmetric tensor representation of matrices to
    flattened representation 

    The input matrices (mat) are of size Nx3x3

    Returns 2D array of flattened symmetric representation 
    of size Nx6
    """

    num_points = mat.shape[0]
    vec = np.zeros((num_points,6))

    vec[:,0] = mat[:,0,0]
    vec[:,1] = mat[:,1,1]
    vec[:,2] = mat[:,2,2]
    vec[:,3] = mat[:,0,1]
    vec[:,4] = mat[:,1,2]
    vec[:,5] = mat[:,2,0]
    
    return vec
