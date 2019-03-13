#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 10:55:54 2018

@author: mbeiran
"""

import numpy as np
import matplotlib.pyplot as plt
import lib_toymodel as ll
import matplotlib as mpl

#### Define the plot parameters

plt.style.use('ggplot')

fig_width = 2.2 # width in inches
fig_height = 2  # height in inches
fig_size =  [fig_width,fig_height]
plt.rcParams['figure.figsize'] = fig_size
plt.rcParams['figure.autolayout'] = True
 
plt.rcParams['lines.linewidth'] = 1.2
plt.rcParams['lines.markeredgewidth'] = 0.003
plt.rcParams['lines.markersize'] = 3
plt.rcParams['font.size'] = 9
plt.rcParams['legend.fontsize'] = 7.
plt.rcParams['axes.facecolor'] = '1'
plt.rcParams['axes.edgecolor'] = '0'
plt.rcParams['axes.linewidth'] = '0.7'

plt.rcParams['axes.labelcolor'] = '0'
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['xtick.color'] = '0'
plt.rcParams['ytick.color'] = '0'
plt.rcParams['xtick.major.size'] = 2
plt.rcParams['ytick.major.size'] = 2

plt.rcParams['font.sans-serif'] = 'Arial'

#%%
#### #### #### #### #### #### #### #### #### #### #### ####
#### Define the system parameters
#### #### #### #### #### #### #### #### #### #### #### ####


phim = 10.
C = 100         #Number of received inputs
g = 4.1        #Inhibition synapse/exc. synapse
f =0.8

CE = C*f
CI = C*(1-f)
gws = np.linspace(0, 10, 200)
Jefs = [-0.2, 0., 0.2]

sol_s = np.zeros((len(gws), len(Jefs)))

sol_find = np.linspace(-10, 10, 100000)

def selff(x, J, CE, CI, g, gw, phim):
    Jef = J*(CE-g*CI)
    return( (Jef-gw)*ll.f_phi(x, phi_max=phim) )
    
def selff_no(x, J, CE, CI, g, gw, phim):
    Jef = J*(CE-g*CI)
    return( Jef*ll.f_phi(x, phi_max=phim)  )

for iJ, J in enumerate(Jefs):
    for ig, gW in enumerate(gws):
        rhs = selff(sol_find, J, CE, CI, g, gW, phim)
        ix = np.argmin(np.abs(sol_find-rhs))
        sol_s[ig, iJ] = sol_find[ix]
        
#%% 
color1= np.array((0.8, 0., 0.))
grayc = np.array((0.5, 0.5, 0.8))

fig = plt.figure()
ax = fig.add_subplot(111)

for iJ, J in enumerate(Jefs):
    plt.plot(gws, ll.f_phi(sol_s[:,iJ], phi_max=phim), color= (2-iJ)*color1/2 + (iJ)*grayc/2, lw= 1., label=r'$J_{eff} =$'+str(J))

plt.legend(frameon=False, loc=1)
plt.ylabel(r'stationary firing rate')
plt.xlabel(r'coupling $g_{w}$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.xlim([0,6])
plt.savefig('adap_fig2_B.pdf')

