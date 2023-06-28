"""
Solve A @ x[i] = c[i] * B @ x[i], the generalised eigenvalue problem for c[i] eigenvalues with 
corresponding eigenvectors x[i] over a range of wavenumbers

Example
-------
python calculate_evals_evecs_multiple.py 150 100 -0.4 NEMO_25 3 1e-8 2e-5 200
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
parser.add_argument('k'          , type=float, help='Zonal wavenumber')
parser.add_argument('iteration'  , type=str  , help='iteration')
args = parser.parse_args()

ny, nz, init_guess, case, values, k, iteration = args.ny, args.nz, args.init_guess, args.case, args.values, args.k, args.iteration
init_guess = init_guess + 0.1*1j

WORK_DIR = os.getcwd() 

# Labels used for the initial guess eigenvalue for the filename
guess_re = str(init_guess.real*100).replace('.','').replace('-','m')

#fname = f'{WORK_DIR}/saved_data/{case}/growth_{values}_{ny:02}_{nz:02}_{str(int(k*1e8))}_{guess_re}.txt'
fname = f'{WORK_DIR}/saved_data/{case}/growth_{values}_{ny:02}_{nz:02}_{iteration}_{guess_re}.txt'
    
cs = eigensolver_multiple.gep(ny, nz, k, case, init_guess, values, tol_input=1e-6)[0]
np.savetxt(fname, cs.view(float).reshape(-1, 2))
print(f'k={k}, cs0={cs[0]}, cs1={cs[1]}, cs2={cs[2]}')
