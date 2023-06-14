"""
Calculate the meshgrids for the half/full points on the staggered grid
"""
import numpy as np
import os

WORK_DIR = os.getcwd()  

def grid(ny, nz, case):
    """
    Calculate the meshgrids for the domain on each grid
 
    Parameters
    ----------
    
    ny : int
         Meridional grid resolution
        
    nz : int
         Vertical grid resolution
        
    case : str
         Mean fields about which to perform the linear stability analysis
         e.g. Proehl_[1-8] - Proehls test cases (Proehl (1996) and Proehl (1998))
              NEMO_25      - Data from the 1/4deg coupled AOGCM
              NEMO_12      - Data from the 1/12deg coupled AOGCM
        
    Returns
    -------
    
    y, y_mid, z, z_mid : (ny, ) array
        Linearly spaced points in the meridional/vertical direction
    
    dy, dz : float
        Meridional/Vertical grid spacing
        
    Y, Y_mid, Y_half, Y_full : 2D array
        Meshgrid with Z, Z_mid, Z_half, Z_full
        
    L : float
        Meridional half-width of the domain
        
    D : float
        Depth of the domain
    """
    
    if case == 'NEMO_25':
        lat = np.loadtxt(f'{WORK_DIR}/NEMO_data/latitude_25.txt')
        
    elif case == 'NEMO_12':
        lat = np.loadtxt(f'{WORK_DIR}/NEMO_data/latitude_12.txt')
        
    else:
        lat = [10]
        
    L = abs(lat[0])*111*1000
    D = 1000
    
    y = np.linspace(-L, L, ny); z = np.linspace(-D, -0, nz) 

    dy = abs(y[1]-y[0]); y_mid = (y[:y.size] + 0.5*dy)[:-1]
    dz = abs(z[1]-z[0]); z_mid = (z[:z.size] + 0.5*dz)[:-1]

    Y,Z         = np.meshgrid(y, z);         Y_full,Z_half = np.meshgrid(y, z_mid) 
    Y_mid,Z_mid = np.meshgrid(y_mid, z_mid); Y_half,Z_full = np.meshgrid(y_mid, z)
    
    return y, y_mid, dy, Y, Y_mid, Y_half, Y_full, z, z_mid, dz, Z, Z_mid, Z_half, Z_full, L, D
