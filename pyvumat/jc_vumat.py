# Simple implementation of Johnson-Cook used for performance benchmarking in 
#   Crone, JC. Computational Materials Science 262 (2026): 114377.
#
# Uses closed-form (non-iterative) return mapping, therefore does
# not capture strain- rate effects.

import sys
import numpy as np

if sys.version_info[0] == 2:
    import ConfigParser as configparser
else: 
    import configparser

from pyvumat.abq_mat_utils import *

#Numerical cutoff
EPSILON = 1.0e-10

class JcVumat:
    def __init__(self,config_file=None):
        pass

    def evaluate(self, **kwargs):

        # Extract required arguments from the keywords
        props = kwargs['props']
        strainInc = kwargs['strainInc']
        stateOld = kwargs['stateOld']
        stressOld = kwargs['stressOld']
        density = kwargs['density']
        tempOld = kwargs['tempOld']
        tempNew = kwargs['tempNew']
        dt = kwargs['dt']
        totalTime = kwargs['totalTime']

        # Loop over material points to compute outputs
        num_points = stressOld.shape[0]
        
        # Initialize return parameters
        enerInternNew = np.copy(kwargs['enerInternOld'])
        enerInelasNew = np.copy(kwargs['enerInelasOld'])
        stateNew = stateOld.copy()
        stressNew = np.zeros(stressOld.shape)

        # Get the material properties
        E = props[0]              # Youngs Modulus
        pois = props[1]           # Poissons Ratio
        A = props[2]              # A param
        B = props[3]              # B param
        n = props[4]              # n param
        C = props[5]              # C param
        m = props[6]              # m param
        t_melt = props[7]         # Melting Temperature
        t_ref = props[8]          # Reference Temperature
        ref_plast_rate = props[9] # Ref Plastic Strain Rate

        # Constants and modulus
        twomu = E / (1.0 + pois)
        thremu = 1.5 * twomu
        sixmu = 3.0 * twomu
        alamda = twomu * (E - twomu) / (sixmu - 2.0 * E)

        # Trial stress assuming all elastic
        trace_E = np.sum(strainInc[:,:3],axis=1).reshape((-1,1))
        trial_stress = stressOld + twomu*strainInc
        trial_stress[:,:3] += alamda*trace_E

        # Deviatoric part of trial stress
        mean_stress = 1.0/3.0*np.sum(trial_stress[:,:3],axis=1).reshape((-1,1))
        trial_stress[:,:3] -= mean_stress

        # von Mises stress
        stress_sqrd = np.power(trial_stress,2)
        
        # Multiply off-diagonal components by 2 to account for
        # Voigt representation
        stress_sqrd[:,3:] *= 2.0
        
        vmises = np.sqrt(1.5*np.sum(stress_sqrd,axis=1))

        # Previous yield stress
        peeq_old = stateOld[:,0]
        yield_old_pe = A + B * np.power(peeq_old,n)
        
        pe_rate_old = stateOld[:,1]
        rel_pe_rate_old = pe_rate_old / ref_plast_rate
        log_term = np.zeros(rel_pe_rate_old.shape)
        np.log(rel_pe_rate_old,
               out=log_term,
               where=(rel_pe_rate_old > 1.0))
        yield_old_rate = 1.0 + C * log_term
        
        eff_temp = (tempOld - t_ref) / (t_melt - t_ref)
        
        temp_m = np.zeros(tempOld.shape)
        np.power(eff_temp, m,
                 out=temp_m,
                 where=(tempOld>t_ref))
        temp_m = np.clip(temp_m,0.0,1.0)
        yield_old_temp = 1.0 - temp_m
        
        yield_old = yield_old_pe * yield_old_rate * yield_old_temp

        # Derivative of yield w.r.t plastic strain
        hard_pe = np.zeros(peeq_old.shape)
        np.power(peeq_old, n-1,
                 out=hard_pe,
                 where=(np.abs(peeq_old)>EPSILON))
        hard_pe *= n * B * yield_old_rate * yield_old_temp
        hard = hard_pe        

        # Check for yield by determining the factor for plasticity
        # Zero for elastic one for yield
        sigdif = vmises - yield_old
        facyld = (sigdif > 0.0)
        
        deqps = np.divide(facyld * sigdif,
                          thremu + hard)

        # Update state variable
        stateNew[:,0] += deqps
        stateNew[:,1] = deqps / dt

        # Update stress
        yieldNew = yield_old + hard * deqps
        factor = np.divide(yieldNew,
                           yieldNew + thremu * deqps)

        stressNew = trial_stress * factor.reshape((-1,1))
        stressNew[:,:3] += mean_stress

        # Update the specific internal energy
        mid_stress = 0.5 * (stressOld + stressNew)
        stress_dot_strain = np.multiply(mid_stress,
                                        strainInc)
        
        # Multiply off-diagonal components by 2 to account for
        # Voigt representation
        stress_dot_strain[:,3:] *= 2.0

        stress_power = np.sum(stress_dot_strain,axis=1)
        enerInternNew += np.divide(stress_power,
                                      density)

        # Update the dissipated inelastic specific energy
        plasticWorkInc = yieldNew * deqps
        enerInelasNew += np.divide(plasticWorkInc,
                                   density)
        
        return stressNew, stateNew, enerInternNew, enerInelasNew
