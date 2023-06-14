"""
Find the most unstable eigenvalue and corresponding eigenvectors of the square matrices A and B.

Solve A @ x[i] = c[i] * B @ x[i], the generalised eigenvalue problem for c[i] eigenvalues with 
corresponding eigenvectors x[i]
"""

import numpy               as np
from   scipy.sparse.linalg import eigs, ArpackNoConvergence
from   scipy               import sparse
from   interruptingcow     import timeout

def evals(A, B, init_guess, values, tol_input):
    """
    Find the most unstable eigenvalue and eigenvector ofthe square matrices A and B.
    
    Solve the generalised eigenvalue problem, A @ x[i] = c[i] * B @ x[i], for c[i] eigenvalues
    and corresponding eigenvectors x[i]
    
    Parameters
    ----------
    A : complex matrix
         Left coefficient matrix of the generalised eigenvalue problem Ax=cBx
    
    B : complex matrix
         Right coefficient matrix of the generalised eigenvalue problem Ax=cBx
     
    init_guess : float
         Initial eigenvalue guess used to search a region with the Arnoldi method
         This guess corresponds to the real part of the phase speed 
    
    Returns 
    -------
    evals : complex
            Eigenvalue of the most unstable mode
    
    evecs : ndarray
            Corresponding eigenvector of the most unstable mode
    
    maxdiffs : float
            Error of the eigenvalue/eigenvector pair
    """
    time_limit = 60*60
    
    try:
    
        with timeout(time_limit, exception=RuntimeError):
                    
            if tol_input == 0:
                evals, evecs = eigs(A, k=values, M=B, which='LI', sigma=init_guess, v0=np.ones(A.shape[0]), tol=tol_input)
                print(f'    | Completed iterations                                           ')
                print(f'    | Eigenvalue converged                                           ')
                
                diffs = A.dot(evecs)-B.dot(evecs)*evals
                maxdiffs = np.linalg.norm(diffs, axis=0, ord=np.inf)
        
                evecs = evecs[:, np.argmax(evals.imag)]/np.linalg.norm(evecs[:, np.argmax(evals.imag)])
                
            else:
                evals = eigs(A, k=values, M=B, which='LI', sigma=init_guess, v0=np.ones(A.shape[0]), tol=tol_input, return_eigenvectors=False)
                evecs = 0
                maxdiffs = 0
            
    except RuntimeError:
    
        if tol_input == 0:
            print(f'    | Eigenvalue did not converge within the time limit')
            print(f'    | Setting eigenvalues to zero')
            evals, evecs, maxdiffs = np.array([0+0*1j]*values), np.empty(A.shape[0])*np.nan, [0.]
            
        else:
            evals, evecs, maxdiffs = np.array([0+0*1j]*values), 0, 0
    
    return evals, evecs, maxdiffs
