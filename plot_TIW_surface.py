"""
Plot the amplitude and phase of an eigenvector
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from   mpl_toolkits.axes_grid1 import make_axes_locatable

from lib import domain
from lib import mean_fields
from lib import perturbations

WORK_DIR = os.getcwd() 

parser = argparse.ArgumentParser()
parser.add_argument('ny'         , type=int  , help='Number of meridional gridpoints')
parser.add_argument('nz'         , type=int  , help='Number of vertical gridpoints')
parser.add_argument('k'          , type=float, help='Zonal wavenumber')
parser.add_argument('case'       , type=str  , help='Cases: NEMO NEMO_rigid_lid Proehl_[1-8]')
args = parser.parse_args()

ny, nz, k, case = args.ny, args.nz, args.k, args.case

# Calculate the grid for a given case and integration
y, y_mid, dy, Y, Y_mid, Y_half, Y_full, z, z_mid, dz, Z, Z_mid, Z_half, Z_full, L, D = domain.grid(ny, nz, case)

# Dimensional values
g, r0, beta = 9.81, 1026, 2.29e-11  

U, U_mid, U_hf, U_fh, Uy, Uy_mid, Uy_hf, Uz, Uz_mid, Uz_hf, r, r_mid, r_hf, ry, ry_mid, ry_hf, rz, rz_mid, rz_hf  = mean_fields.on_each_grid(ny, nz, case)

# Calculate perturbations
u, u_v, u_w, v, v_p, v_w, p, p_v, p_w, w, w_v, w_p, rho, rho_v, rho_p, cs = perturbations.on_each_grid(ny, nz, k, case)

## PLOT TIWs ############################################################################################################################

def TIW_surface_lat_lon(data,k, depth):

    x = np.linspace(0, 50*111*1000, 201)

    data_cos = data.real[depth, :]
    data_cos = np.repeat(data_cos[:, np.newaxis], 201, axis=1)
    data_cos = data_cos*np.cos(k*x)
    
    data_sin = data.imag[depth, :]
    data_sin = np.repeat(data_sin[:, np.newaxis], 201, axis=1)
    data_sin = -data_sin*np.sin(k*x)
    
    data_TIW = data_cos + data_sin
    
    return data_TIW
    
def TIW_surface_dep_lon(data,k,lat):

    x = np.linspace(0, 50*111*1000, 201)

    data_cos = data.real[:, lat]
    data_cos = np.repeat(data_cos[:, np.newaxis], 201, axis=1)
    data_cos = data_cos*np.cos(k*x)
    
    data_sin = data.imag[:, lat]
    data_sin = np.repeat(data_sin[:, np.newaxis], 201, axis=1)
    data_sin = -data_sin*np.sin(k*x)
    
    data_TIW = data_cos + data_sin
    
    return data_TIW
    
def TIW_surface_dep_lat(data,k,lon):

    x = np.linspace(0, 50*111*1000, 201)

    data_cos = data.real
    data_cos = np.repeat(data_cos[:, :, np.newaxis], 201, axis=2)
    data_cos = data_cos*np.cos(k*x)
    
    data_sin = data.imag
    data_sin = np.repeat(data_sin[:, :, np.newaxis], 201, axis=2)
    data_sin = -data_sin*np.sin(k*x)
    
    data_TIW = (data_cos + data_sin)[:,:,lon]
    
    return data_TIW
    
def TIW_ALL(data, k):

    x = np.linspace(0, 50*111*1000, 201)

    data_cos = data.real
    data_cos = np.repeat(data_cos[:, :, np.newaxis], 201, axis=2)
    data_cos = data_cos*np.cos(k*x)
    
    data_sin = data.imag
    data_sin = np.repeat(data_sin[:, :, np.newaxis], 201, axis=2)
    data_sin = -data_sin*np.sin(k*x)
    
    data_TIW = data_cos + data_sin
    
    return data_TIW

u_all = TIW_ALL(u, k); u_max = np.amax(abs(u_all))
v_all = TIW_ALL(v_p, k); v_max = np.amax(abs(v_all))
p_all = TIW_ALL(p, k); p_max = np.amax(abs(p_all))
r_all = TIW_ALL(rho_p, k); r_max = np.amax(abs(r_all))
w_all = TIW_ALL(w_p, k); w_max = np.amax(abs(w_all))
    
fig = plt.figure(figsize=(12, 6), dpi=300)

ax1 = plt.subplot(221)

x = np.linspace(0, 50*111*1000, 201); nx=len(x)

X, Y = np.meshgrid(x, y_mid)
    
u_TIW = TIW_surface_lat_lon(u,k,-1)
v_TIW = TIW_surface_lat_lon(v_p,k,-1)
p_TIW = TIW_surface_lat_lon(p,k,-1)
r_TIW = TIW_surface_lat_lon(rho_p,k,-1)
w_TIW = TIW_surface_lat_lon(w_p,k,-1)

x_step = 2; y_step = 3

contourf = ax1.contourf(X, Y, r_TIW, levels = r_max*np.linspace(-1, 1, 50), cmap='RdBu_r')
ax1.contour(X, Y, p_TIW, levels = p_max*np.delete(np.linspace(-1, 1, 11), 5) , colors='k', linewidths=.75)

q = ax1.quiver(X[::y_step,::x_step], Y[::y_step,::x_step], u_TIW[::y_step,::x_step], v_TIW[::y_step,::x_step], angles='xy', scale=10, width=0.0015, headlength=3.75, color='k')

#ax1.set_xlim([0*111*1000, 50*111*1000])
#ax1.set_xticks([0*111*1000, 10*111*1000, 20*111*1000, 30*111*1000, 40*111*1000, 50*111*1000])
#ax1.set_xticklabels(['0', '10', '20', '30', '40', '50'])

ax1.set_xlim([0*111*1000, 30*111*1000])
ax1.set_xticks([0*111*1000, 10*111*1000, 20*111*1000, 30*111*1000])
ax1.set_xticklabels(['0', '10', '20', '30'])

ax1.set_ylim([-5*111*1000, 7*111*1000])
ax1.set_yticks([-5*111*1000, 0, 5*111*1000])
ax1.set_yticklabels(['-5', '0', '5'])

ax1.set_xlabel(f'Longitude [deg E]')
ax1.set_ylabel(f'Latitude [deg N]')


ax2 = plt.subplot(223)

x = np.linspace(0, 50*111*1000, 201); nx = len(x)

X, Z = np.meshgrid(x, z_mid)

u_TIW = TIW_surface_dep_lon(u, k, 85)
v_TIW = TIW_surface_dep_lon(v_p, k, 85)
w_TIW = TIW_surface_dep_lon(w_p, k, 85)
r_TIW = TIW_surface_dep_lon(rho_p, k, 85)
p_TIW = TIW_surface_dep_lon(p, k, 85)

x_step, z_step = 3, 1

contourf = ax2.contourf(X, Z, r_TIW, levels = r_max*np.linspace(-1, 1, 50), cmap='RdBu_r')

ax2.contour(X, Z, v_TIW, levels = v_max*np.delete(np.linspace(-1, 1, 21), 10) , colors='k', linewidths=.75)

q = ax2.quiver(X[::z_step,::x_step], Z[::z_step,::x_step], u_TIW[::z_step,::x_step], w_TIW[::z_step,::x_step], angles='xy', scale=3, width=0.002, headlength=3.75)

#ax2.set_xlim([0*111*1000, 50*111*1000])
#ax2.set_xticks([0*111*1000, 10*111*1000, 20*111*1000, 30*111*1000, 40*111*1000, 50*111*1000])
#ax2.set_xticklabels(['0', '10', '20', '30', '40', '50'])

ax2.set_xlim([0*111*1000, 30*111*1000])
ax2.set_xticks([0*111*1000, 10*111*1000, 20*111*1000, 30*111*1000])
ax2.set_xticklabels(['0', '10', '20', '30'])

ax2.set_ylim([-150, 0])
ax2.set_yticks([-150, -100, -50, 0])
ax2.set_yticklabels([f'150', f'100', f'50', f'0'])

ax2.set_xlabel(f'Longitude [deg E]')
ax2.set_ylabel(f'Depth [m]')

ax3 = plt.subplot(122)

Y, Z = np.meshgrid(y_mid, z_mid); y_step, z_step = 1, 1; longitude = 58; print(f'Longitude={x[longitude]/(1000*111)} deg E')

u_TIW = TIW_surface_dep_lat(u, k, longitude)
v_TIW = TIW_surface_dep_lat(v_p, k, longitude)
w_TIW = TIW_surface_dep_lat(w_p, k, longitude)
r_TIW = TIW_surface_dep_lat(rho_p, k, longitude)
p_TIW = TIW_surface_dep_lat(p, k, longitude)

contourf = ax3.contourf(Y, Z, r_TIW, levels = r_max*np.linspace(-1, 1, 100), cmap='RdBu_r')
ax3.contour(Y, Z, u_TIW, levels = u_max*np.delete(np.linspace(-1, 1, 21), 10) , colors='k', linewidths=0.75)

#ax1.axvline(x=x[longitude])
#ax1.axhline(y=y_mid[85])

q = ax3.quiver(Y[::z_step,::y_step], Z[::z_step,::y_step], v_TIW[::z_step,::y_step], w_TIW[::z_step,::y_step], angles='xy', scale=6, width=0.002, headlength=4)

ca      = contourf.axes
fig     = ca.figure
divider = make_axes_locatable(ca)
cax     = divider.append_axes("right", size="5%", pad=0.05)
cbar    = fig.colorbar(contourf, cax=cax, label=f'', ticks=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])

ax3.set_xlim([-5*111*1000, 7*111*1000])
ax3.set_ylim([-150, 0])

ax3.set_xticks([-5*111*1000, 0, 5*111*1000])
ax3.set_xticklabels(['-5', '0', '5'])

ax3.set_yticks([-150, -100, -50, 0])
ax3.set_yticklabels([f'150', f'100', f'50', f'0'])

ax3.set_xlabel(f'Latitude [deg N]')
ax3.set_ylabel(f'Depth [m]')

plt.tight_layout()
plt.savefig(f'/home/rees/MRes/images/Chapter_5/3D_Visualisation_{case}_{int(k*1e8)}.png')
plt.show()
