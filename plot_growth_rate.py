"""
Plot growth rates
"""

import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('ny'         , type=int  , help='Number of meridional gridpoints')
parser.add_argument('nz'         , type=int  , help='Number of vertical gridpoints')
parser.add_argument('case'       , type=str  , help='Cases: NEMO NEMO_rigid_lid Proehl_[1-8]')
parser.add_argument('values'     , type=int  , help='Number of output eigenvalues')
args = parser.parse_args()

ny, nz, case, values = args.ny, args.nz, args.case, args.values

WORK_DIR = os.getcwd() 

k_wavenum = np.linspace(1e-8, 2e-5, 150); k_end = 2e-5; k_num = 150

fname = f'{WORK_DIR}/saved_data/{case}/growth_{values}_{ny:02}_{nz:02}_m*.txt'
    
#fname = f'{WORK_DIR}/saved_data/NEMO_25/u-bx950/unstable/RAW/growth_rates/growth_3_150_100_*.txt'   
    
files = sorted(glob.glob(fname)); #print(files)

cs = np.array([np.loadtxt(filename).view(complex).reshape(values, k_num) for filename in files])# (7, 3, 150)

color = ['deeppink','orange','red','green','steelblue','darkviolet','black']
#color = ['red','green','steelblue','darkviolet','black']

fig, axes=plt.subplots(figsize=(6,4))

#axes2 = axes.twinx()

for i in range(len(files)):

    cs0 = cs[i,:,:]; sigma = np.asarray([abs(k_wavenum[i])*cs0[:, i].imag for i in range(k_num)])

    axes.plot(k_wavenum, sigma, '.', ms=3, color=color[i])

axes.set_xlabel(r'$k$ [m$^{-1}$]', fontsize=18)
axes.set_ylabel(r'Growth Rate [s$^{-1}$]', fontsize=18)

axes.set_xlim([0, k_end])
axes.set_ylim([0, 6e-7])
axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
axes.tick_params(axis='both', which='major', labelsize=16)
#axes2.tick_params(axis='both', which='major', labelsize=16)

#growth_day = lambda sigma: sigma*86400
#ymin, ymax = axes.get_ylim()
#axes2.set_ylim((growth_day(ymin), growth_day(ymax)))
#axes2.plot([],[])

axes.grid(alpha=.5)

axes.axvspan((2*np.pi/(2000*1000)), (2*np.pi)/(500*1000), color='k', alpha=.5)

plt.tight_layout()
plt.savefig(f'/home/rees/MRes/images/Chapter_5/growth_rate_{case}.png', dpi=300, bbox_inches='tight')
plt.show()
