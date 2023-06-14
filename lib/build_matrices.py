"""
Build the matrices A and B for the generalised eigenvalue problem Ax=cBx
"""

import numpy               as np
from   scipy.integrate     import trapz
from   scipy.linalg        import block_diag
from   scipy.sparse        import diags
from   scipy               import sparse

from lib import domain
from lib import mean_fields 

def tridiag(subdiag, diag, supdiag, bc):
    """
    Create a tridiagonal matrix of the form
    
    |  bc[0]           bc[1]           0              0           0            ...               .                   0          |
    |subdiag[0]      diag[1]       supdiag[1]         0           0            ...               .                   0          |
    |    0          subdiag[1]      diag[2]       supdiag[2]      0            ...               .                   0          |
    |    0               0         subdiag[2]      diag[3]    supdiag[3]       ...               .                   0          |
    |    .               .             .              .           .            ...               .                   .          |
    |    .               .             .              .           .            ...               .                   .          |
    |    .               .             .              .           .            ...               .                   0          |
    |    0               .             .              .           .       subdiag[-2]       diag[-2]           supdiag[-1]      |
    |    0               0             0              0           0             0              bc[2]               bc[3]        |
    
    This allows us to easily implement the vertical boundary conditions
                        
    Parameters
    ----------
    
    subdiag : float
        Elements on the subdiagonal of the tridagonal matrix
        
    diag : float
        Elements on the diagonal of the tridiagonal matrix
        
    supdiag : float
        Elements on the superdiagonal of the tridiagonal matrix
        
    bc : list of floats
        Entries corresponding to boundary conditions
        
    Returns
    -------
    
    tri_matrix : matrix
        Tridiagonal matrix 
    """
    
    # Create a diagonal matrix with tridiagonal entries and then convert to a regular numpy array
    tri_matrix = diags([subdiag, diag, supdiag],[-1, 0, 1]).toarray()
    
    # Add entries corresponding to boundary conditions
    tri_matrix[0, 0]   = bc[0]; tri_matrix[0, 1]   = bc[1]
    tri_matrix[-1, -2] = bc[2]; tri_matrix[-1, -1] = bc[3]
    
    return tri_matrix
 
def gep(ny, nz, k, case): 
    """
    Construct the matrices A and B for the generalised eigenvalue problem (gep) Ax=cBx
                            
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
    A : complex matrix
        Left coefficient matrix of the generalised eigenvalue problem Ax=cBx
    
    B : complex matrix
        Right coefficient matrix of the generalised eigenvalue problem Ax=cBx
    """
    
    ########################################################################################################################################################################################################
    # (I) Set up the domain 
    ########################################################################################################################################################################################################

    # Calculate the grid for a given case and integration
    y, y_mid, dy, Y, Y_mid, Y_half, Y_full, z, z_mid, dz, Z, Z_mid, Z_half, Z_full, L, D = domain.grid(ny, nz, case)
        
    ########################################################################################################################################################################################################
    # (II) Specify the mean zonal flow and denisty profiles 
    ########################################################################################################################################################################################################

    U, U_mid, U_hf, U_fh, Uy, Uy_mid, Uy_hf, Uz, Uz_mid, Uz_hf, r, r_mid, r_hf, ry, ry_mid, ry_hf, rz, rz_mid, rz_hf = mean_fields.on_each_grid(ny, nz, case)

    ########################################################################################################################################################################################################
    # (III) Typical dimensional parameters 
    ########################################################################################################################################################################################################
           
    # Typical values for the equatorial ocean 
    beta   = 2.29e-11          # Meridional gradient of the Coriolis parameter (m^{-1}s^{-1})
    r0     = 1026              # Background density (kg m^{3})
    g      = 9.81              # Gravitational acceleration (ms^{-2})

    ########################################################################################################################################################################################################
    # (IV) Build the zonal momentum equation 
    ########################################################################################################################################################################################################
                
    ZLU = k*np.diag(U_mid.flatten(order='F'))

    ZLV0 = np.zeros(((ny-1)*(nz-1), (ny*(nz-1)))) 

    for i in range((ny-1)*(nz-1)):                
        ZLV0[i, i]        = beta*Y_mid.flatten(order='F')[i]                          
        ZLV0[i, i+(nz-1)] = beta*Y_mid.flatten(order='F')[i] 

    ZLV1 = np.zeros(((ny-1)*(nz-1), (ny*(nz-1)))) 
    for i in range((ny-1)*(nz-1)):                
        ZLV1[i, i]        = (Uy_mid.flatten(order='F'))[i]                            
        ZLV1[i, i+(nz-1)] = (Uy_mid.flatten(order='F'))[i] 

    ZLV2 = np.zeros(((ny-1)*(nz-1), (ny*(nz-1))))  
    for i in range(ny-1):                        
        ZLV2[i*(nz-1):(i+1)*(nz-1), i*(nz-1):(i+1)*(nz-1)] = tridiag((ry_hf/rz_hf)[1:-1, i], ((ry_hf/rz_hf)[:-1, i]+(ry_hf/rz_hf)[1:, i]), (ry_hf/rz_hf)[1:-1, i], [(ry_hf/rz_hf)[1, i], (ry_hf/rz_hf)[1, i], (ry_hf/rz_hf)[-2, i], (ry_hf/rz_hf)[-2, i]]) 
        ZLV2[i*(nz-1):(i+1)*(nz-1), i*(nz-1)+(nz-1):(i+1)*(nz-1)+(nz-1)] = tridiag((ry_hf/rz_hf)[1:-1, i], ((ry_hf/rz_hf)[:-1, i]+(ry_hf/rz_hf)[1:, i]), (ry_hf/rz_hf)[1:-1, i], [(ry_hf/rz_hf)[1, i], (ry_hf/rz_hf)[1, i], (ry_hf/rz_hf)[-2, i], (ry_hf/rz_hf)[-2, i]]) 

    ZLV2 = (Uz_mid.flatten(order='F')*(ZLV2.T)).T 

    ZLV = (1j/2)*ZLV0 - (1j/2)*ZLV1 + (1j/8)*ZLV2 

    ZLP0 = (k/r0)*np.eye((ny-1)*(nz-1))

    ZLP1 = np.asarray([tridiag((U_hf/rz_hf)[1:-1, i], -((U_hf/rz_hf)[:-1, i]-(U_hf/rz_hf)[1:, i]), -(U_hf/rz_hf)[1:-1, i], [(U_hf/rz_hf)[1, i], -(U_hf/rz_hf)[1, i], (U_hf/rz_hf)[-2, i], -(U_hf/rz_hf)[-2, i]]) for i in range(ny-1)])
    ZLP1 = block_diag(*ZLP1) 
    ZLP1 = (Uz_mid.flatten(order='F')*(ZLP1.T)).T

    ZLP = ZLP0 - (1/g)*(k/(2*dz))*ZLP1

    ZRU = k*np.eye((ny-1)*(nz-1))
    ZRV = np.zeros_like(ZLV)

    ZRP1 = np.asarray([tridiag((1/rz_hf)[1:-1, i], -((1/rz_hf)[:-1, i]-(1/rz_hf)[1:, i]), -(1/rz_hf)[1:-1, i], [(1/rz_hf)[1, i], -(1/rz_hf)[1, i], (1/rz_hf)[-2, i], -(1/rz_hf)[-2, i]]) for i in range(ny-1)])
    ZRP1 = block_diag(*ZRP1) 
    ZRP1 = (Uz_mid.flatten(order='F')*(ZRP1.T)).T

    ZRP = np.zeros_like(ZLP0) - (1/g)*(k/(2*dz))*ZRP1

    ########################################################################################################################################################################################################
    # (V) Build the meridional momentum equation 
    ########################################################################################################################################################################################################

    MLU0 = np.zeros((ny*(nz-1), (ny-1)*(nz-1)))
    for i in range((ny-2)*(nz-1)):                
        MLU0[i+(nz-1), i]        = beta*Y_full.flatten(order='F')[i+(nz-1)]                 
        MLU0[i+(nz-1), i+(nz-1)] = beta*Y_full.flatten(order='F')[i+(nz-1)] 
    MLU = (-1j/2)*MLU0
        
    MLV = k*np.diag(U_fh.flatten(order='F'))

    MLP0 = np.zeros((ny*(nz-1), (ny-1)*(nz-1))) 
    for i in range((ny-2)*(nz-1)):                
        MLP0[i+(nz-1), i]        = 1                              
        MLP0[i+(nz-1), i+(nz-1)] = -1 
    MLP = (1/r0)*(1j/dy)*MLP0
    
    MRU = np.zeros((ny*(nz-1), (ny-1)*(nz-1)))
    MRV = k*np.eye(ny*(nz-1)) 
    MRP = np.zeros((ny*(nz-1), (ny-1)*(nz-1)))

    ########################################################################################################################################################################################################
    # (VI) Build the continuity equation (CHECKED)
    ########################################################################################################################################################################################################   
    
    CLU = k*np.eye((ny-1)*(nz-1))

    CLV0 = np.zeros(((ny-1)*(nz-1), (ny*(nz-1)))) 
    for i in range((ny-1)*(nz-1)):               
        CLV0[i, i]        = 1                           
        CLV0[i, i+(nz-1)] = -1

    CLV1 = np.zeros(((ny-1)*(nz-1), (ny*(nz-1)))) 
    for i in range(ny-1):                        
        CLV1[i*(nz-1):(i+1)*(nz-1), i*(nz-1):(i+1)*(nz-1)] = tridiag((ry_hf/rz_hf)[1:-1, i], ((ry_hf/rz_hf)[:-1, i]-(ry_hf/rz_hf)[1:, i]), -(ry_hf/rz_hf)[1:-1, i], [-(ry_hf/rz_hf)[1, i], -(ry_hf/rz_hf)[1, i], (ry_hf/rz_hf)[-2, i], (ry_hf/rz_hf)[-2, i]]) 
        CLV1[i*(nz-1):(i+1)*(nz-1), i*(nz-1)+(nz-1):(i+1)*(nz-1)+(nz-1)] = tridiag((ry_hf/rz_hf)[1:-1, i], ((ry_hf/rz_hf)[:-1, i]-(ry_hf/rz_hf)[1:, i]), -(ry_hf/rz_hf)[1:-1, i], [-(ry_hf/rz_hf)[1, i], -(ry_hf/rz_hf)[1, i], (ry_hf/rz_hf)[-2, i], (ry_hf/rz_hf)[-2, i]]) 

    CLV = (1j/dy)*CLV0 - (1j/(4*dz))*CLV1 

    CLP_blocks = np.asarray([tridiag((U_hf/rz_hf)[1:-1, i], -((U_hf/rz_hf)[:-1, i]+(U_hf/rz_hf)[1:, i]), (U_hf/rz_hf)[1:-1, i], [-(U_hf/rz_hf)[1, i], (U_hf/rz_hf)[1, i], (U_hf/rz_hf)[-2, i], -(U_hf/rz_hf)[-2, i]]) for i in range(ny-1)])
    CLP = (1/g)*(k/(dz**2))*block_diag(*CLP_blocks) 

    CRU = np.zeros_like(CLU)
    CRV = np.zeros_like(CLV)

    CRP_blocks = np.asarray([tridiag((1/rz_hf)[1:-1, i], -((1/rz_hf)[:-1, i]+(1/rz_hf)[1:, i]), (1/rz_hf)[1:-1, i], [-(1/rz_hf)[1, i], (1/rz_hf)[1, i], (1/rz_hf)[-2, i], -(1/rz_hf)[-2, i]]) for i in range(ny-1)])
    CRP = (1/g)*(k/(dz**2))*block_diag(*CRP_blocks) 
        
    ########################################################################################################################################################################################################
    # (VII) Apply boundary conditions 
    ########################################################################################################################################################################################################

    # No-normal flow at the meridional walls of the domain
    MLV[:(nz-1),:(nz-1)]   = np.diag(np.ones(nz-1)) 
    MLV[-(nz-1):,-(nz-1):] = np.diag(np.ones(nz-1))

    MRV[:(nz-1),:]  = 0
    MRV[-(nz-1):,:] = 0

    ########################################################################################################################################################################################################
    # (VIII) Build the coefficient matrices A and B of the generalised eigenvalue problem 
    ########################################################################################################################################################################################################
    
    # Form the LHS and RHS of each equation of motion as matrices
    ZLE = np.hstack([ZLU, ZLV, ZLP]); MLE = np.hstack([MLU, MLV, MLP]); CLE = np.hstack([CLU, CLV, CLP])
    ZRE = np.hstack([ZRU, ZRV, ZRP]); MRE = np.hstack([MRU, MRV, MRP]); CRE = np.hstack([CRU, CRV, CRP])
    
    # Build the coefficient matrices 
    A = np.vstack([ZLE, MLE, CLE]); B = np.vstack([ZRE, MRE, CRE])

    
    return A, B
