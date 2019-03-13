#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 13:52:03 2018

@author: mbeiran
"""

import numpy as np
import matplotlib.pyplot as plt
import lib_toymodel as ll

#### Define the plot parameters

plt.style.use('ggplot')

fig_width = 2.2 # width in inches
fig_height = 2  # height in inches
fig_size =  [fig_width,2*fig_height]
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

center_color = np.array(( 	176/255., 196/255., 222/255.))
color1= np.array((0.8, 0., 0.))
center_color1 = 0.85*center_color+0.15*color1
center_color2 = 0.6*center_color+0.4*color1

x = np.linspace(1.0, 40.0, 10000)
B = x-1.
T=0.5
lw = 1.5
fracs = [0.2,0.2]
lls = []
fracS = np.linspace(0,10, 800)

#%%
fig = plt.figure()

for i, gS in enumerate([ 0.5]):#, 0.0005,0.005,([0.2,0.02, 0.002, 0.0002]):
    
    ax = fig.add_subplot(2,1,i+1)
    
    ax.set_ylabel( r"adaptive time ratio $\frac{\tau}{\tau_w}$" )
    NN = len(fracs)
    
    borderT = np.sqrt(2*gS/(1+gS))
    
    
    
    its = 1.0*len(lls)
    gg = np.linspace(0, 4.0, 100)
    rr = np.linspace(0, 2.5, 100)
    
    oms = np.linspace(0,2, 2000)
    trans1 = np.zeros(len(fracS))
    omC = np.zeros(len(fracS))

    for i, f in enumerate(fracS):
        trans1[i] = np.min(np.abs((-oms**2+1j*(1+f)*oms+f*(1+gS))/(f+1j*oms)))
        omC[i] = oms[np.argmin(np.abs((-oms**2+1j*(1+f)*oms+f*(1+gS))/(f+1j*oms)))]
    trans1[0]=1.
    plt.plot(trans1[fracS<borderT], fracS[fracS<borderT], '--', linewidth=1.5, c='k')
    plt.plot(trans1[fracS>borderT], fracS[fracS>borderT], '-', linewidth=1.5, c='k')
    
    x = np.linspace(0,3)
    y = 0.5*np.ones(len(x))
            
    ax.fill_between(np.hstack((0,trans1[1:])), np.hstack((-1, fracS[1:])), 10, 
                    color=center_color2)
    ax.fill_between(np.hstack((trans1[1:],10)), np.hstack(( fracS[1:], 10)), -10, 
                    color=center_color1)
    
    ax.plot([0,10], fracs, '-.', linewidth=0.1, color='k')
    ax.plot([1.3, 1.3], [0, 1], '-.', linewidth=0.1, color='k')
    ax.set_xlim([0., 2.0])
    ax.set_ylim([0.0, 1.])
    
xx = trans1[fracS<borderT]
yy = fracS[fracS<borderT]

xx = xx[-1]
yy = yy[-1]

ax.scatter([0.95, 1.05, 1.19, 1.43], [0.2, 0.2, 0.2, 0.2],  10,   facecolors='none', edgecolors= 'k', lw=0.6)
ax.scatter([1.3,  1.3], [1/1.68,  1/31.],  10,  marker='^', facecolors='none', edgecolors= 'k', lw=0.6)


ax.scatter([xx], [yy], 20,  facecolors='k', edgecolors= [1, 1, 1])
ax.set_xticks([0,   1])
ax.set_xticklabels(['0', '1'])
ax.set_yticks([0,  1])
ax.set_yticklabels(['0',  '1'])
#ax.plot( [-0.05,0.05], [alpha,alpha] ,'k', linewidth = 2)
#ax.text(0.1,T-0.07, r'$\alpha$', fontsize=25)
   
ax.text(0.25,0.65, 'Stable')
#ax.text(1.6,0.6, 'chaos' , fontsize=25)
#ax.text(1.22,0.328,'oscill.', fontsize=25)
horpoints = [  0.8, 1.0,  1.1, 1.15, 1.2, 1.3, 1.4,  1.45, 1.5, 1.6, 1.7, 1.8]
verpoints = np.logspace(0.1, 1.5,12)


gs = np.linspace(0.01, 5, 100)

#%%
rMin = 0.03
rMax = 10.0
frac=0.3

it=0
itmax = 25
tol = 0.000001



def func_bisect(r, frac, g_w):
    ts = np.arange(0.00001,2*np.pi, 0.001)
    xs_m = r*np.exp(1j*ts)
    lambda1_m, lambda2_m = ll.eig_limitcurve(xs_m, frac, g_w)
    sym_m = np.max(np.max(np.real(lambda1_m)), np.max(np.real(lambda2_m)))        
    return(sym_m)

def func_minIm(r, frac, g_w):
    ts = np.arange(0.00001,2*np.pi, 0.001)
    xs_m = r*np.exp(1j*ts)
    lambda1_m, lambda2_m = ll.eig_limitcurve(xs_m, frac, g_w)
    lambd = np.hstack((lambda1_m, lambda2_m))

    lambd = np.abs(np.imag(lambd[np.real(lambd)>0]))
    
    
    if lambd.size == 0:
        sym_m = 1.0
    else:
        sym_m = np.min(lambd)
    return(sym_m)

lls = []
st_chaos = []
st_osc = []
om_osc = []
gS = np.linspace(0,10, 800)
for i, f in enumerate(fracs):
    sT_chaos =ll.bound_chaos(gS, f) 
    st_chaos.append(sT_chaos)
    sT_osc, omC = ll.bound_osc(gS, f)
    st_osc.append(sT_osc)
    om_osc.append(omC)
#
#center_color = np.array(( 	176/255., 196/255., 222/255.))
#color1= np.array((0.8, 0., 0.))
#center_color1 = 0.85*center_color+0.15*color1
#
#center_color2 = 0.6*center_color+0.4*color1
ax = fig.add_subplot(212)
ax.set_xlabel(r"connectivity strength $J\sqrt{C_E+g^2 C_I}$")
ax.set_ylabel( r"adaptive coupling $g_w$")

NN = len(fracs)

gborder = frac**2/(2-frac**2)
for i, f in enumerate(fracs[:-1]):
    x = st_osc[i]
    x1 = x[gS<gborder]
    x2 = x[gS>gborder]
    ax.plot(x1, gS[gS<gborder],  '-', label=r'$T=$'+str(fracs[i]), color='k', lw=1.5)
    ax.plot(x2, gS[gS>gborder],  '--', label=r'$T=$'+str(fracs[i]), color='k', lw=1.5)


xx = x1[-1]
yy = gS[gS<gborder]
yy = yy[-1]

    
ax.plot([0, 10], [0.5, 0.5], '-.', linewidth=0.1, color='k')
ax.fill_between(np.hstack((0,st_osc[i])), np.hstack((-1, gS)), 10, 
                color=center_color2)


ax.fill_between(np.hstack((st_osc[i],10)), np.hstack(( gS, 10)), -10, 
                color=center_color1)

ax.scatter([xx], [yy], 20, facecolors='k', edgecolors= [1, 1, 1])

ax.scatter([0.95, 1.05, 1.19,  1.43], [0.5, 0.5, 0.5, 0.5 ],  10,   facecolors='none', edgecolors= 'k', lw=0.6)

ax.scatter([1.3], [0.5],  10, marker='^',  facecolors='none', edgecolors= 'k', lw=0.6)

ax.text(0.25,1.0, 'Stable')
ax.set_xlim([0.0, 1.9])
ax.set_ylim([0.0, 2.0])
ax.set_xticks([0,   1])
ax.set_xticklabels(['0', '1'])
ax.set_yticks([0,  1, 2])
ax.set_yticklabels(['0',  '1', '2'])

plt.savefig('adap_fig3_C.pdf')
plt.show()

#%%
fig_size =  [fig_width, fig_height]
plt.rcParams['figure.figsize'] = fig_size

fig = plt.figure()
for i, gS in enumerate([ 0.5]):#, 0.0005,0.005,([0.2,0.02, 0.002, 0.0002]):
    
    ax = fig.add_subplot(1,1,i+1)
    
    ax.set_xlabel ( "connectivity strength $J\sqrt{C_E+g^2 C_I}$")
    ax.set_ylabel( r"synaptic time ratio $\frac{\tau}{\tau_s}$" )
    xl   = " "
    yl   = r"synaptic time ratio $\frac{\tau}{\tau_s}$" 
    if i==0 or i==2:
        yl = ylab
    if i>1:
        xl = xlab
    #ll.get_rightframe(ax, majory=1.0, minory=0.5 , majorx=1.0, minorx=10.5, fontsize = 25, \
    #xlabel=xl, ylabel=yl, labelsize = 25, foursides = True,
    #lenm=10., lenmi=5., widthm=1., widthmi=1.)
    NN = len(fracs)
    
    its = 1.0*len(lls)
    gg = np.linspace(0, 4.0, 100)
    rr = np.linspace(0, 2.5, 100)
    
    oms = np.linspace(0,2, 2000)
    trans1 = np.zeros(len(fracS))
    omC = np.zeros(len(fracS))

    for i, f in enumerate(fracS):
        trans1[i] = np.min(np.abs((-oms**2+1j*(1+f)*oms+f*(1+gS))/(f+1j*oms)))
        omC[i] = oms[np.argmin(np.abs((-oms**2+1j*(1+f)*oms+f*(1+gS))/(f+1j*oms)))]
    trans1[0]=1.
    plt.plot(trans1/trans1, fracS, '-', linewidth=1.5, c='k')
    x = np.linspace(0,3)
    y = 0.5*np.ones(len(x))
        
    #plt.plot(bound_chaos(gS, fracS), fracS, '--', linewidth=2, c='k')
    center_color = np.array(( 	176/255., 196/255., 222/255.))
    color1= np.array((0.8, 0., 0.))
    center_color1 = 0.85*center_color+0.15*color1
    
    center_color2 = 0.2*center_color+0.8*0.9*np.ones(3)
    ax.fill_between(np.hstack((0,trans1[1:]/trans1[1:])), np.hstack((-1, fracS[1:])), 10, 
                    color=center_color)
    ax.fill_between(np.hstack((1,10)), 0, 10, color=center_color2)
    
    #ax.fill_between(np.hstack((trans1[1:]/trans[1:],10)), np.hstack(( fracS[1:], 10)), -10, 
    #                color=center_color1)
    
    #ax.fill_between(np.hstack((0,trans1)), np.hstack((-1,gS)), 10, color=[ 	135/255., 206/255., 250/255.])
       
    #ax.text(1.9, 0.7, r'$g_w=$'+str(gS), fontsize=20)
    #ax.scatter([1.6,],[0.2, ], s=100)
    ax.plot([0,10], [0.2, 0.2], '-.', linewidth=0.1, color='k')
    ax.plot([1.3, 1.3], [0, 1], '-.', linewidth=0.1, color='k')
    ax.set_xlim([0., 2.0])
    ax.set_ylim([0.0, 1.])
    

ax.scatter([0.95, 1.05, 1.19, 1.43], [0.2, 0.2, 0.2, 0.2],  10,   facecolors='none', edgecolors= 'k', lw=0.6)
ax.set_xticks([0,   1])
ax.set_xticklabels(['0', '1'])
ax.set_yticks([0,  1])
ax.set_yticklabels(['0',  '1'])
   
ax.text(0.25,0.65, 'Stable')
#ax.text(1.6,0.6, 'chaos' , fontsize=25)
#ax.text(1.22,0.328,'oscill.', fontsize=25)
horpoints = [  0.8, 1.0,  1.1, 1.15, 1.2, 1.3, 1.4,  1.45, 1.5, 1.6, 1.7, 1.8]
verpoints = np.logspace(0.1, 1.5,12)


plt.savefig('adap_fig3_D.pdf')
plt.show()
