#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 11:34:08 2018

@author: mbeiran
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import lib_toymodel as ll

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

#### #### #### #### #### #### #### #### #### #### #### ####
#### Define the system parameters
#### #### #### #### #### #### #### #### #### #### #### ####
JS =0

Tau_ws = 1.1*np.logspace(0,2.5, 1000)
g_w = 5.0

tau = 1.0
C = 100         #Number of received inputs
g = 4.1   
f = 0.8   
J = 0.5/(np.sqrt(C*f+g**2*C*(1-f)))#3.7    #J0=0.045 with the parameters

Jvar = np.sqrt(J**2*(C*f+g**2*C*(1-f)))
J  = J*(C*f-g*C*(1-f))


color1= np.array((0.8, 0., 0.))
color_end = np.array((0.3, 0.3, 0.3))

    
def eigvals_adap(g_w, tau_w, tau, J):
    T = tau/tau_w;
    M = (1./tau)*np.array(((J-1, -g_w),(T, -T)))
    eigvs, R = np.linalg.eig(M)
    dR = np.linalg.det(R)
    amp1 = (1./dR)*R[1,1]*R[0,0]
    amp2  = -(1./dR)*R[0,1]*R[1,0]
    return(eigvs, amp1, amp2)
#%%

  
tconst = np.zeros(len(Tau_ws))
tconst2 = np.zeros(len(Tau_ws))
tconst1 = np.zeros(len(Tau_ws))

syn1 = np.ones(len(Tau_ws))
syn2 = Tau_ws     
grayc = [0.3, 0.3, 0.3]

for it, taw in enumerate(Tau_ws):
    eigvs, amp1, amp2 = eigvals_adap(g_w, taw, tau, J)
    ts = -1./np.real(eigvs)
            
    tconst[it] = (np.abs(amp1)*ts[0] + np.abs(amp2)*ts[1])/np.sum(np.abs(amp1)+np.abs(amp2))
    tconst1[it] = 1./np.abs(np.real(eigvs[0]))
    tconst2[it] = 1./np.abs(np.real(eigvs[1]))

tm = 1./(-(JS-1-2.*g_w)-np.sqrt((JS-1-2.*g_w)**2-(JS-1)**2))
ix = np.argmin(np.abs(tm-Tau_ws))



fig = plt.figure()
ax = fig.add_subplot(111)
#plt.plot(Tau_ws, tconst1, color = 'C'+str(iJ),  label='J='+str(JS))
#plt.plot(Tau_ws, tconst2, color = 'C'+str(iJ))
#plt.plot(Tau_ws, tconst, '--',color = 'C'+str(iJ))
plt.plot(Tau_ws, syn2, '-',color = grayc)
plt.plot(Tau_ws, syn1, '-',color = grayc)
plt.plot(Tau_ws, tconst1, color = color1, label='Adaptive')
plt.plot(Tau_ws, tconst2, color = color1)
plt.plot(Tau_ws, tconst, '--',color = color1)
    

#plt.plot(Tau_ws, syn1, '-',color = grayc, label='Synaptic filter')
#plt.plot(Tau_ws[::30], syn1[::30], '.',color = 'k', label='Synaptic filter')


plt.plot(Tau_ws, 0.5*syn2+0.5*syn1, '--',color = grayc)
plt.scatter([5, ], [0.8, ],20, marker='v',   facecolors='none', edgecolors= [0.1, 0.1, 0.1], lw=1.2)
plt.scatter([30, ], [0.8, ],20, marker='v',   facecolors='none', edgecolors= [0.6, 0.6, 0.6], lw=1.2)
plt.xscale('log')
plt.yscale('log')
plt.ylim([0.7, 100])
plt.xlim([0.95, 300])
plt.xlabel(r'slower time constant $\tau_{w/s}$')
plt.ylabel(r'activity timescales ')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.savefig('adap_fig1_B.pdf')
