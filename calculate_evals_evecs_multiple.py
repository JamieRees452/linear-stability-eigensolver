"""
Find the eigenvalues and corresponding eigenvectors of the square matrices A and B with the largest imaginary parts
over a range of wavenumbers

Solve A @ x[i] = c[i] * B @ x[i], the generalised eigenvalue problem for c[i] eigenvalues with 
corresponding eigenvectors x[i]

Example
-------
python calculate_evals_evecs_multiple.py 50 50 -0.2 NEMO_25 01 02 1e-8 1e-5 50
"""

import argparse
import numpy as np
import os
import sys
from   tqdm import tqdm

from lib import eigensolver_multiple

parser = argparse.ArgumentParser()
parser.add_argument('ny'         , type=int  , help='Number of meridional gridpoints')
parser.add_argument('nz'         , type=int  , help='Number of vertical gridpoints')
parser.add_argument('init_guess' , type=float, help='initial guess for the eigenvalue')
parser.add_argument('case'       , type=str  , help='Cases: NEMO_25, NEMO_12, Proehl_[1-8]')
parser.add_argument('values'     , type=int  , help='Number of output eigenvalues')
parser.add_argument('k_start'    , type=float, help='Starting wavenumber')
parser.add_argument('k_end'      , type=float, help='Ending wavenumber')
parser.add_argument('k_num'      , type=int  , help='Number of steps')
args = parser.parse_args()

ny, nz, init_guess, case, values, k_start, k_end, k_num = args.ny, args.nz, args.init_guess, args.case, args.values, args.k_start, args.k_end, args.k_num
init_guess = init_guess + 0.1*1j

WORK_DIR = os.getcwd() 

# Labels used for the initial guess eigenvalue for the filename
guess_re = str(init_guess.real*100).replace('.','').replace('-','m')

# Set up the range of wavenumbers to calculate growth rates over
k_wavenum = np.linspace(k_start, k_end, k_num)
cs = np.zeros((values, k_num), dtype=complex) # Initialise the storage of the eigenvalues

fname = f'{WORK_DIR}/saved_data/{case}/growth_{values}_{ny:02}_{nz:02}_{guess_re}.txt'

if os.path.exists(fname):

    user_input = input(f'{fname}\n\nFile already exists\nOverwrite existing saved data? (y/n)')
    
    if user_input == 'y':
    
        print(f'Overwriting existing saved data. Calculating eigenvalues over a range of wavenumbers...')
        
        for count, k in enumerate(tqdm(k_wavenum, position=0, leave=True)):
        
            cs[:, count] = eigensolver_multiple.gep(ny, nz, k, case, init_guess, values, tol_input=1e-6)[0]
            np.savetxt(fname, cs.view(float).reshape(-1, 2))
        
    else:
    
        print(f'Previously calculated eigenvalues are located at {fname}')
else:

    print(f'Calculating eigenvalues over a range of wavenumbers...\n\nSaving eigenvalues to {fname}')
    
    for count, k in enumerate(tqdm(k_wavenum, position=0, leave=True)):
    
        cs[:, count] = eigensolver_multiple.gep(ny, nz, k, case, init_guess, values, tol_input=1e-6)[0]
        np.savetxt(fname, cs.view(float).reshape(-1, 2))
