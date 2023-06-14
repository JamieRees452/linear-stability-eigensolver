"""
Calculate/obtain the mean zonal velocity and density fields, from NEMO data or from Proehls test cases, on each grid
e.g. Zonal velocity (U) at (h)alf points in y, (f)ull points in z is given by U_hf
"""

import numpy as np
import os
from   scipy import integrate
from   scipy.integrate import trapz
import sys

from lib import calculate_NEMO_fields
from lib import calculate_Proehl_fields
from lib import domain

def on_each_grid(ny, nz, case):
    """
    Calculate the mean fields on each grid
                            
    Parameters
    ----------
    ny : int
         Meridional grid resolution
    
    nz : int
         Vertical grid resolution
     
    k : float
         Zonal wavenumber
    
    init_guess : float
         Initial eigenvalue guess used to search a region with the Arnoldi method
         This guess corresponds to the real part of the phase speed 
         
    case : str
         Mean fields about which to perform the linear stability analysis
         e.g. Proehl_[1-8] - Proehls test cases (Proehl (1996) and Proehl (1998))
              NEMO_25      - Data from the 1/4deg coupled AOGCM
              NEMO_12      - Data from the 1/12deg coupled AOGCM
        
    Returns
    -------
    U : (nz, ny) ndarray
         Mean zonal velocity
    
    r : (nz, ny) ndarray
         Mean density field
        
    U_mid, U_hf : ndarray
         Mean zonal velocity calculated at different points on the staggered grid
         
    r_mid, r_hf, r_fh : ndarray
         Mean density calculated at different points on the staggered grid
         
    Uy, Uy_mid, Uy_hf, ry, ry_mid, ry_hf : ndarray
         Meridional gradients of mean zonal velocity and density at different points on the staggered grid
         
    Uz, Uz_mid, Uz_hf, rz, rz_mid, rz_hf : ndarray
         Vertical gradients of mean zonal velocity and density at different points on the staggered grid
    """

    # Calculate the grid for a given case and integration
    y, y_mid, dy, Y, Y_mid, Y_half, Y_full, z, z_mid, dz, Z, Z_mid, Z_half, Z_full, L, D = domain.grid(ny, nz, case)
    
    beta   = 2.29e-11          # Meridional gradient of the Coriolis parameter (m^{-1}s^{-1})
    r0     = 1026              # Background density (kg m^{3})
    g      = 9.81              # Gravitational acceleration (ms^{-2})
    
    # Obtain the mean zonal velocity fields
    
    if (case == 'Proehl_1' or case == 'Proehl_2' or case == 'Proehl_3' or case == 'Proehl_4' or
        case == 'Proehl_5' or case == 'Proehl_6' or case == 'Proehl_7' or case == 'Proehl_8'):
        
        U    , Uy    , Uz     = calculate_Proehl_fields.mean_velocity(Y     , Z     , case)
        U_mid, Uy_mid, Uz_mid = calculate_Proehl_fields.mean_velocity(Y_mid , Z_mid , case)
        U_hf , Uy_hf , Uz_hf  = calculate_Proehl_fields.mean_velocity(Y_half, Z_full, case)
        U_fh , Uy_fh , Uz_fh  = calculate_Proehl_fields.mean_velocity(Y_full, Z_half, case)

        N2     = 8.883e-5*np.ones(Z.shape[0])
        N2_mid = 8.883e-5*np.ones(Z_mid.shape[0])

        r = (r0/g)*(beta*integrate.cumtrapz(Y*Uz, y, initial=0) - np.tile(integrate.cumtrapz(N2, z, initial=0), (len(y), 1)).T) + r0
        ry = (beta*r0/g)*Y*Uz; rz = np.gradient(r, z, axis=0); 

        r_hf = (r0/g)*(beta*integrate.cumtrapz(Y_half*Uz_hf, y_mid, initial=0) - np.tile(integrate.cumtrapz(N2, z, initial=0), (len(y)-1, 1)).T) + r0
        ry_hf = (beta*r0/g)*Y_half*Uz_hf; rz_hf = np.gradient(r_hf, z, axis=0)

        r_mid = (r0/g)*(beta*integrate.cumtrapz(Y_mid*Uz_mid, y_mid, initial=0) - np.tile(integrate.cumtrapz(N2_mid, z_mid, initial=0), (len(y)-1, 1)).T) + r0
        ry_mid = (beta*r0/g)*Y_mid*Uz_mid; rz_mid = np.gradient(r_mid, z_mid, axis=0)
        
    elif case == 'NEMO_25' or case == 'NEMO_12':
    
        U, U_mid, U_hf, U_fh, Uy, Uy_mid, Uy_hf, Uz, Uz_mid, Uz_hf = calculate_NEMO_fields.load_mean_velocity(ny, nz, case)
        r, r_mid, r_hf, ry, ry_mid, ry_hf, rz, rz_mid, rz_hf       = calculate_NEMO_fields.load_mean_density(ny, nz, case)
        
    else:
        print(f'{case} in not a valid case')
        
    return U, U_mid, U_hf, U_fh, Uy, Uy_mid, Uy_hf, Uz, Uz_mid, Uz_hf, r, r_mid, r_hf, ry,ry_mid, ry_hf, rz, rz_mid, rz_hf 
