"""
Find the most unstable eigenvalue and corresponding eigenvectors of the square matrices A and B.

Solve A @ x[i] = c[i] * B @ x[i], the generalised eigenvalue problem for c[i] eigenvalues with 
corresponding eigenvectors x[i]
"""

from lib import build_matrices
from lib import solve_gep

def gep(ny, nz, k, init_guess, case, values, tol_input):
    """
    Find the most unstable eigenvalue and eigenvector ofthe square matrices A and B.
    
    Solve the generalised eigenvalue problem, A @ x[i] = c[i] * B @ x[i], for c[i] eigenvalues
    and corresponding eigenvectors x[i]
    
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
    evals : complex
            Eigenvalue of the most unstable mode
    
    evecs : ndarray
            Corresponding eigenvector of the most unstable mode
    
    maxdiffs : float
            Error of the eigenvalue/eigenvector pair
    """
    
    print(f'    | Building matrices A and B                                      ')
    A, B = build_matrices.gep(ny, nz, k, case)
    print(f'    | Successfully built A and B                                     ')

    print(f'    | Solve using implicitly restarted Arnoldi method                ')
    print(f'    | Starting iterations                                            ')
    evals, evecs, maxdiffs = solve_gep.evals(A, B, init_guess, values, tol_input)
      
    return evals, evecs, maxdiffs
