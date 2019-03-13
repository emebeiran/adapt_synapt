#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 15:10:18 2018

@author: mbeiran
"""
import numpy as np
import matplotlib.pyplot as plt
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
#%%

fig = plt.figure()
ax = fig.add_subplot(111)

plt.xlabel(  r"adaptation time $\tau_w$" )
plt.ylabel( r"Hopf frequency $\omega_C$" )
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
tauWs = np.linspace(0.01,1, 800)
color1 = np.array((0.8, 0, 0))
color2 = 0.8*np.ones(3)
color0 = 0.2*np.ones(3)
colS = [0.5*color1+0.5*color0, color1, 0.5*color1+0.5*color2, color2]  
g_ws = [ 2., 0.2, 0.02] 
for i, g_w in enumerate(g_ws):
    sT_osc, omC = ll.bound_oscfrac( g_w, tauWs)
    plt.plot(1./tauWs, omC,  color=colS[i], label=r"$g_w =$"+str(g_w))
    tauL = 1./(g_w+np.sqrt(2*g_w*(g_w+1)))
    print(tauL)
    if tauL>1.:
        plt.scatter([tauL], [0.02],  10,   marker = 'o', facecolors=colS[i], edgecolors= 'k', lw=0.6, zorder =20)
ax.set_ylim([0,1.15])
ax.set_xlim([-0.5,100])
plt.legend( loc=1)

#plt.yticks([0, 5, 10])
plt.xticks([1, 25, 50, 75, 100])

plt.savefig('adap_fig3_E.pdf')
#plt.xlim([0, 8])
plt.show()
