#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 17:17:08 2018

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

T  = 100
dt = 0.01 
time = np.arange(0, T, dt)


g_w = 5.0
taus = [5,30]
gamma = 0.5


color1s = np.zeros((3,2))
graycs = np.zeros((3,2))
color1s[0,0] = 0.8
graycs[:,0]  = 0.3*np.ones(3)
color1s[:,1] = 0.5*(color1s[:,0]+np.ones(3))
graycs[:,1]  = 0.5*(graycs[:,0]+np.ones(3))

xa = np.zeros((len(time), 2))
xs = np.zeros((len(time), 2))
xn = np.zeros((len(time), 2))
filta = np.zeros((len(time),2))

filts = np.zeros((len(time),2))
filts = np.zeros((len(time),2))
filt1 = np.zeros((len(time),2))
stim = np.zeros((len(time),1))
stim[0] = 1/dt

def obtain_trace(x, index=0):
    y = x[:,index]
    return(y/np.max(y))
def obtain_trace2(x, index=0):
    y = x[:,index]
    return(y)    
    
#### #### #### #### #### #### #### #### #### #### #### ####
#### Plot
#### #### #### #### #### #### #### #### #### #### #### ####

fig = plt.figure()
ax = plt.axes(frameon=True)

for iv, vals in enumerate(taus):
    tau_w = vals
    tau_s = vals
    
    for i, t in enumerate(time[:-1]):

        
        #xn[i+1,0] = xn[i,0] + dt*(-xn[i,0])+np.sqrt(dt)*s[i]

        filta[i+1,0] = filta[i,0] + dt*(-filta[i,0] -g_w*filta[i,1])+stim[i]
        filta[i+1,1] = filta[i,1] + dt*(-filta[i,1]+filta[i,0])/tau_w
        
        filts[i+1,0] = filts[i,0] + dt*(-filts[i,0] +filts[i,1])
        filts[i+1,1] = filts[i,1] + (dt*(-filts[i,1])+stim[i])/tau_s
        
        #filt1[i+1,0] = filt1[i,0] + dt*(-filt1[i,0]+stim[i])
    
    filtA    = obtain_trace(filta)
    filtS    = obtain_trace(filts)
    #filtOne  = obtain_trace(filt1)

  
    plt.plot(time,filtA,  label = 'Adaptive', color = color1s[:,iv])
    plt.plot(time, filtS,  label = 'Synaptic', color= graycs[:,iv])

    if iv==0:
        plt.legend(frameon=False)
        
plt.ylabel('normalized filter')
plt.xlabel('time lag')
ax.set_ylim([-0.25,1.3])
ax.set_xlim([-1, 25])
ax.plot([-1, 25], [0,0], c='k', lw=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=5)
plt.savefig('adap_fig1_A.pdf')
plt.show()

#%%
np.random.seed(2)
T  = 300
dt = 0.01 
time = np.arange(0, T, dt)


g_w = 5.0
gamma = 0.5


color1s = np.zeros((3,2))
graycs = np.zeros((3,2))
color1s[0,0] = 0.8
graycs[:,0]  = 0.3*np.ones(3)
color1s[:,1] = 0.5*(color1s[:,0]+np.ones(3))
graycs[:,1]  = 0.5*(graycs[:,0]+np.ones(3))

xa = np.zeros((len(time), 2))
xs = np.zeros((len(time), 2))
xn = np.zeros((len(time), 2))
filta = np.zeros((len(time),2))

filts = np.zeros((len(time),2))
filts = np.zeros((len(time),2))
filt1 = np.zeros((len(time),2))
stim = np.zeros((len(time),1))

#tau_stim = 1.5;
#for i,t in enumerate(time[:-1])
#    stim[i+1] = stim[i] + (dt/tau_stim)*(-stim[i]+np.sqrt(dt)*np.random.randn(1)/tau_stim)
t0 = 100

for vals in range(200):
    stim += np.cos(2.5*np.random.rand(1,1)*time+2*np.pi*np.random.rand(1,1)).T
stim = stim/np.sqrt(200)

stim[time<110]=0
stim[time>190]=0
fig = plt.figure()
ax = plt.axes(frameon=True)

for iv, vals in enumerate(taus):
    tau_w = vals
    tau_s = vals
    
    for i, t in enumerate(time[:-1]):

        
        #xn[i+1,0] = xn[i,0] + dt*(-xn[i,0])+np.sqrt(dt)*s[i]

        filta[i+1,0] = filta[i,0] + dt*(-filta[i,0] -g_w*filta[i,1])+stim[i]
        filta[i+1,1] = filta[i,1] + dt*(-filta[i,1]+filta[i,0]+gamma)/tau_w
        
        filts[i+1,0] = filts[i,0] + dt*(-filts[i,0] +filts[i,1])
        filts[i+1,1] = filts[i,1] + (dt*(-filts[i,1])+stim[i])/tau_s
        
        #filt1[i+1,0] = filt1[i,0] + dt*(-filt1[i,0]+stim[i])
    
    filtA    = obtain_trace2(filta)
    filtS    = obtain_trace2(filts)
    #filtOne  = obtain_trace(filt1)

    if iv==0:
        plt.plot(time-t0,filtA+1200,  label = 'Adaptive', color = color1s[:,iv])
        plt.plot(time-t0, filtS+600,  label = 'Synaptic', color= graycs[:,iv])
    else:
        plt.plot(time-t0,filtA+900,  label = 'Adaptive', color = color1s[:,iv])
        plt.plot(time-t0, filtS+300,  label = 'Synaptic', color= graycs[:,iv])
    print(np.max(filtA))
    print(np.max(filtS))
    
#    if iv==0:
#        plt.legend(frameon=False)

plt.plot(time-t0, 100*stim-2., 'k')
plt.text(0, 1270, r'$\tau_w = $'+str(taus[0]), fontsize=7.)
plt.text(0, 970, r'$\tau_w = $'+str(taus[1]), fontsize=7.)
plt.text(0, 670, r'$\tau_s = $'+str(taus[0]), fontsize=7.)
plt.text(0, 370, r'$\tau_s = $'+str(taus[1]), fontsize=7.)
plt.text(0, 120, 'stimulus', fontsize=7.)



#plt.ylabel('normalized filter')
plt.xlabel('time')
#ax.set_ylim([-0.25,1.3])
ax.set_xlim([0, 100])
#ax.plot([-1, 25], [0,0], c='k', lw=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.yaxis.set_visible(False)
ax.xaxis.set_ticks_position('bottom')

plt.locator_params(nbins=1)
plt.savefig('adap_fig1_C.pdf')
plt.show()
