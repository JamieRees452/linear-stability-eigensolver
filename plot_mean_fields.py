"""
Plot the mean fields U, Uy, Uz, r, ry, rz
"""

import argparse
import matplotlib
import matplotlib.pyplot   as plt
import numpy               as np
import os 
import sys
from   mpl_toolkits.axes_grid1 import make_axes_locatable

from lib import calculate_NEMO_fields
from lib import domain
from lib import mean_fields

parser = argparse.ArgumentParser()
parser.add_argument('ny'         , type=int  , help='Number of meridional gridpoints')
parser.add_argument('nz'         , type=int  , help='Number of vertical gridpoints')
parser.add_argument('case'       , type=str  , help='Cases: NEMO NEMO_rigid_lid Proehl_[1-8]')
args = parser.parse_args()

ny, nz, case = args.ny, args.nz, args.case

WORK_DIR = os.getcwd() 

# Calculate the grid for a given case and integration
y, y_mid, dy, Y, Y_mid, Y_half, Y_full, z, z_mid, dz, Z, Z_mid, Z_half, Z_full, L, D = domain.grid(ny, nz, case)

# Calculate the mean fields on each grid
U, U_mid, U_hf, U_fh, Uy, Uy_mid, Uy_hf, Uz, Uz_mid, Uz_hf, r, r_mid, r_hf, ry, ry_mid, ry_hf, rz, rz_mid, rz_hf = mean_fields.on_each_grid(ny, nz, case)

# Set the dimensional parameters
g, r0, beta = 9.81, 1026, 2.29e-11; dy = abs(y[1]-y[0])   

Q  = -(1/r0)*(ry*Uz + (beta*Y-Uy)*rz)
Qy = np.gradient(Q, y, axis=1)      

N2 = -(g/r0)*rz 

fig, axes=plt.subplots(figsize=(12,8), nrows=2, ncols=2, sharex=True, sharey=True)

contourf = axes[0,0].contourf(Y, Z, U, levels = np.delete(np.linspace(-1.5, 1.5, 31), 15), cmap='RdBu_r')
axes[0,0].contour(Y, Z, U, levels = np.delete(np.linspace(-1.5, 1.5, 31), 15), colors='k', linewidths=0.5)
axes[0,0].contour(Y, Z, Qy, levels=[0], colors='k')

ca      = contourf.axes
fig     = ca.figure
divider = make_axes_locatable(ca)
cax     = divider.append_axes("right", size="5%", pad=0.05)
cbar    = fig.colorbar(contourf, cax=cax, label=f'', ticks=[-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5])

contourf = axes[0,1].contourf(Y, Z, r, levels = np.linspace(1021.0, 1028.0, 15), cmap='viridis_r')
axes[0,1].contour(Y, Z, r, levels = np.linspace(1021.5, 1027.5, 13), colors='k', linewidths=0.5)
axes[0,1].contour(Y, Z, Qy, levels=[0], colors='k')

ca      = contourf.axes
fig     = ca.figure
divider = make_axes_locatable(ca)
cax     = divider.append_axes("right", size="5%", pad=0.05)
cbar    = fig.colorbar(contourf, cax=cax, label=f'', ticks=[1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028])
cbar.ax.invert_yaxis()

contourf = axes[1,0].contourf(Y, Z, Qy, levels = 5e-15*np.delete(np.linspace(-1.0, 1.0, 21), 10), cmap='RdBu_r')
axes[1,0].contour(Y, Z, Qy, levels = 5e-15*np.delete(np.linspace(-1.0, 1.0, 21), 10), colors='k', linewidths=0.5)
axes[1,0].contour(Y, Z, Qy, levels=[0], colors='k')

ca      = contourf.axes
fig     = ca.figure
divider = make_axes_locatable(ca)
cax     = divider.append_axes("right", size="5%", pad=0.05)
cbar    = fig.colorbar(contourf, cax=cax, label=f'', ticks=[-5e-15, -2.5e-15, 0, 2.5e-15, 5e-15])

contourf = axes[1,1].contourf(Y, Z, N2, levels = np.linspace(0, 6e-4, 25), cmap='viridis')
axes[1,1].contour(Y, Z, N2, levels = np.linspace(0, 6e-4, 12), colors='k', linewidths=0.5)
axes[1,1].contour(Y, Z, Qy, levels=[0], colors='k')

ca      = contourf.axes
fig     = ca.figure
divider = make_axes_locatable(ca)
cax     = divider.append_axes("right", size="5%", pad=0.05)
cbar    = fig.colorbar(contourf, cax=cax, label=f'')#, ticks=[-5e-15, -2.5e-15, 0, 2.5e-15, 5e-15])
cbar.formatter.set_powerlimits((0, 0))

if case == 'Proehl_1' or case == 'Proehl_2':    
    axes[0].set_xlim([-3e5, 0])
    axes[0].set_ylim([-800, -200])

    axes[0].set_xticks([-3e5, -2e5, -1e5, 0])
    axes[0].set_yticks([-800, -500, -200])
    
    axes[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
elif case == 'Proehl_3':
    axes[0].set_xlim([-1e6, 0])
    axes[0].set_ylim([-300, 0])

    axes[0].set_xticks([-1e6, -8e5, -6e5, -4e5, -2e5, 0])
    axes[0].set_yticks([-300, -150, 0])
    
    axes[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
elif (case == 'Proehl_4' or case == 'Proehl_5' or case == 'Proehl_6' or
     case == 'Proehl_7' or case == 'Proehl_8'):
     
    axes[0].set_xlim([-8e5, 8e5])
    axes[0].set_ylim([-250, 0])

    axes[0].set_xticks([-8e5, -6e5, -4e5, -2e5, 0, 2e5, 4e5, 6e5, 8e5])
    axes[0].set_yticks([-250, -200, -150, -100, -50, 0])
    
    axes[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
else:
    
    axes[0,0].set_xlim([-111*10*1000, 111*10*1000])
    axes[0,0].set_ylim([-250, 0])

    axes[0,0].set_xticks([-10*111*1000, -5*111*1000, 0, 5*111*1000, 10*111*1000])
    axes[0,0].set_yticks([-250, -200, -150, -100, -50, 0])
    
    axes[0,0].set_xticklabels(['-10', '-5', '0', '5', '10'])
    axes[0,0].set_yticklabels(['250', '200', '150', '100', '50', '0'])

axes[0,0].tick_params(axis='both', which='major', labelsize=16)
axes[0,1].tick_params(axis='both', which='major', labelsize=16)
axes[1,0].tick_params(axis='both', which='major', labelsize=16)
axes[1,1].tick_params(axis='both', which='major', labelsize=16)

axes[1,0].set_xlabel(f'Latitude [deg N]', fontsize=16)
axes[1,1].set_xlabel(f'Latitude [deg N]', fontsize=16)

axes[0,0].set_ylabel(f'Depth [m]', fontsize=16)
axes[1,0].set_ylabel(f'Depth [m]', fontsize=16)

axes[0,0].set_title(r'(a)', fontsize=16)
axes[0,1].set_title(r'(b)', fontsize=16)
axes[1,0].set_title(r'(c)', fontsize=16)
axes[1,1].set_title(r'(d)', fontsize=16)

plt.tight_layout()
#plt.savefig(f'/home/rees/MRes/images/Chapter_5/mean_fields_{case}.png', dpi=300, bbox_inches='tight')
plt.show()
