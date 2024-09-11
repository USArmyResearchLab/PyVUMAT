import sys
import numpy as np
if sys.version_info[0] == 2:
    import ConfigParser as configparser
else: 
    import configparser
from pyvumat.abq_mat_utils import *

class SvkVumat:
    def __init__(self,config_file=None):
 
        # Default function for computing stress
        self.stress_func = compute_stress_abq_vec
        
        if config_file is not None:
            # Parse the options from the ini file
            parser = configparser.ConfigParser()
            parser.read(config_file)

            if parser.has_option('Model','StressFunction'):
            
                # Chose the function to compute stress
                func_name = parser.get('Model','StressFunction')
                if func_name == 'loop':
                    self.stress_func = compute_stress_loop
                elif func_name == 'matrix_vectorize':
                    self.stress_func = compute_stress_mat_vec
                elif func_name == 'abaqus_vectorize':
                    self.stress_func = compute_stress_abq_vec
                else:
                    print("\n Error: unknown function name \n")
                    sys.stdout.flush()
                    sys.exit()

    def evaluate(self, **kwargs):

        # Extract required arguments from the keywords
        props = kwargs['props']
        stretchNew = kwargs['stretchNew']
        stateOld = kwargs['stateOld']
        
        # Get the elastic properties
        youngs_mod = props[0]
        nu = props[1]
        two_mu = youngs_mod/(1+nu)
        lame1 = two_mu*nu/(1.0-2.0*nu)
        
        # Create storage for return values
        num_points = stretchNew.shape[0]
        stateNew = stateOld.copy()
        enerInternNew = np.zeros(num_points)
        enerInelasNew = np.zeros(num_points)

        # Compute the stress using the implementation 
        # specified in the configuration file
        stressNew = self.stress_func(stretchNew,
                                     lame1,
                                     two_mu)
        
        # Update dummy state variables
        stateNew[:,0] += 1.0
        stateNew[:,1] -= 1.0

        return stressNew, stateNew, enerInternNew, enerInelasNew

def compute_stress_loop(stretch, lame1, two_mu):

    # Storage for temporary matricies
    num_points = stretch.shape[0]
    I = np.identity(3)
    U = np.zeros((3,3))
    E = np.zeros((3,3))
    S = np.zeros((3,3))
    stress_mat = np.zeros((3,3))
    stress = np.zeros((num_points,6))
    
    # Loop through material points and compute stress
    for i in range(num_points):
        U[0,0], U[1,1] = stretch[i,0], stretch[i,1]
        U[2,2], U[0,1] = stretch[i,2], stretch[i,3]
        U[1,2], U[2,0] = stretch[i,4], stretch[i,5]
        U[1,0], U[2,1], U[0,2] = U[0,1], U[1,2], U[2,0]
        
        E = 0.5*(np.dot(U,U) - I)
        S = lame1*np.trace(E)*I + two_mu*E
        stress_mat = np.dot(np.dot(U,S),U.T)/np.linalg.det(U)
        
        stress[i,0], stress[i,1] = stress_mat[0,0], stress_mat[1,1]
        stress[i,2], stress[i,3] = stress_mat[2,2], stress_mat[0,1]
        stress[i,4], stress[i,5] = stress_mat[1,2], stress_mat[2,0]

    return stress
        
def compute_stress_mat_vec(stretch, lame1, two_mu):

    # Put stretch tensor (U) in [N, 3, 3] array
    U = abq_vec_to_mat(stretch)
    
    # Compute Green-Lagrange strain (E = 1/2 (U.U - I)
    I = np.identity(3).reshape(1,3,3)
    E = 0.5 * (np.matmul(U,U) - I)
    trace_E = np.trace(E, axis1=1, axis2=2).reshape(-1,1,1)
    
    # compute 2nd PK stress (S)
    S = two_mu*E  +  lame1*trace_E*I
    
    # Convert from 2nd PK (S) to corotational Cauchy = U.S.U/J
    det_U = np.linalg.det(U).reshape((-1,1))
    stress_mat = np.matmul(np.matmul(U,S),U)    
    
    return abq_mat_to_symm_vec(stress_mat)*(1.0/det_U)

def compute_stress_abq_vec(stretch, lame1, two_mu):

    # Compute Green-Lagrange strain E = 1/2 (U.U - I)
    E = abq_mat_mult(stretch,stretch)
    E[:,:3] -= 1.0
    E *= 0.5
    trace_E = np.sum(E[:,:3], axis=1).reshape((-1,1))
    
    # Compute 2nd PK stress (S)
    stress = two_mu*E
    stress[:,:3] += lame1*trace_E
    
    # Convert from 2nd PK (S) to corotational Cauchy = U.S.U/J
    J = abq_det(stretch).reshape(-1,1)
    cauchy = abq_mat_mult(abq_mat_mult(stretch,stress),stretch)/J
    
    return cauchy[:,:6]
