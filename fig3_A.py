# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 16:10:39 2016

@author: mbeiran
"""

import numpy as np
import matplotlib.pylab as plt
import lib_toymodel as ll
import matplotlib as mpl

from scipy.optimize import fsolve


#### Define the plot parameters

plt.style.use('ggplot')

fig_width = 2.2 # width in inches
fig_height = 2  # height in inches
fig_size =  [3*fig_width,fig_height]
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

#Parameters

N = 3000        #Number of neurons
C = 100         #Number of received inputs
g = 4.1         #Inhibition synapse/exc. synapse
f =0.8

fig = plt.figure()
for ifact, fact in enumerate([0.95, 1.09, 1.175, 1.4]):
    J = fact/(np.sqrt(C*f+g**2*C*(1-f)))#3.7    #J0=0.045 with the parameters
    phi_max = 2.0
    g_w = 0.5
    frac = 0.2  #Should be smaller than 1
    
    T = 180
    dt = 0.05
    time = np.arange(0,T,dt)
    
    #%  Find fix point
    def my_func(x):
        y = ll.f_phi(x, phi_max=phi_max)*J*(f*C-g*(1-f)*C)-x
        return(y)
        
    try:     
        fl = np.load("matrix_data_J"+str(fact)+".npz")
        Jmat = fl['name1']
        w = fl['name2']
        sol = fl['name3']
        sol2 = fl['name4']
    except:
        print('inside 1')
        try:
            fl = np.load("matrix_data_J"+str(fact)+".npz")
            Jmat = fl['name1']
            w = fl['name2']
        except:
            print('inside2')
            np.random.seed(10)
            unstable = True
            un = False 
            counter = 0
            if unstable:
                while ~un:
                    print(counter)
                    counter = counter + 1
                    ss = np.random.randint(10000)
                    Jmat = J*ll.create_Jmat(N, C, g=g, seed = ss)
                    Q = np.zeros((2*N,2*N))
                    Q[0:N,0:N]=Jmat-np.eye(N)
                    Q[0:N,N:]=-g_w*np.eye(N)
                    Q[N:,0:N]=frac*np.eye(N)
                    Q[N:,N:]=-frac*np.eye(N)
                    
            
            
                    CE = f*C
                    CI = (1-f)*C
                    x0 = fsolve(my_func, 0.0)
                    r = J*np.sqrt(CE*f+g**2*CI) 
                    J0 = 1.0/(ll.phi_der(x0, phi_max=phi_max)*np.sqrt(C*f+g**2*(1-f)*C)) #Critical J
                    outlier = J*(CE-g*CI) #outlier of the eigenvalue
                    #%
                    
                    w = np.linalg.eigvals(Q)
                    if counter > 15 or ifact!=2:
                        un = True
                        break
                        print('Max count')
                    
                    if np.max(np.real(w))>0 and np.sum(np.real(w)>0)<3 and ifact==2:
                        un = True
                        break
                    print(np.sum(np.real(w)>0))
        xws0 = 1.*np.random.rand(2*N)
       
        phimax = 2.
        tau = 1.
        tau_w = 5.
        sol, sol2, meanr, meana, q1, q2 = ll.rate_simfinal(time, xws0, Jmat, tau, tau_w, g_w, phimax, method='rk4', numbs =2*int(N))
        np.savez("matrix_data_J"+str(fact), name1 = Jmat, name2 = w, name3= sol, name4 = sol2)

    #%%
    r = J*np.sqrt(C*f+g**2*C*(1-f))
    thet = np.linspace(0, 2*np.pi, 1000)
    
    point1 = -0.5*(frac+1)-0.5*np.sqrt((frac+1)**2-4*(frac)*(1+g_w)+0j)
    point2 = -0.5*(frac+1)+0.5*np.sqrt((frac+1)**2-4*(frac)*(1+g_w)+0j)
    
                
    def func_transf(x, frac, g_w):
        trace = (frac+1-x+0j)**2-4*frac*(1+g_w-x+0j)
        fix = -frac-1+x
        lambda1 = 0.5*(fix+np.sqrt(trace))
        lambda2 = 0.5*(fix-np.sqrt(trace))
        return(lambda1, lambda2)
        
    def func_aprox(x, frac, g_w):
        lambda1 = -0.5*frac*(1+((1-x+2*g_w)/(1-x)))
        lambda2 = -1+x+frac*g_w*(1-np.conj(x))/(np.abs(1-x))**2
        return(lambda1, lambda2)
        
    ts = np.arange(0.001,2*np.pi, 0.001)
    xs = r*np.exp(1j*ts)
    lambda1, lambda2 = func_transf(xs, frac, g_w)
    l_ap1, l_ap2 = func_aprox(xs, frac, g_w)
    
    cluster1=-frac*(1+g_w)*(xs/(1-xs))+point2
    XL = [-2.7, 0.7]
             
    #%% Rate model
   
    dt = 0.05
    time = np.arange(0,T,dt)
    tau = 1.
    tau_w = 5.
    
    

    ax = fig.add_subplot(1,4,ifact+1)
    xlab = "time"
    if ifact==0:
        ax.set_ylabel("firing rate")
    
    color1= np.array((0.8, 0., 0.))
    
    nums = 4
    if ifact==0:
        colors = np.zeros((3, nums))
    for ind, n in enumerate(range(nums)):
        if ifact==0:
            rnd = (np.random.rand(3))
            colors[:,n] = 0.2*color1 + 0.8*rnd
        ax.plot(time, ll.f_phi(sol[:, ind], phi_max=phi_max), color=colors[:,n], lw = 0.5)
        
    mean_act = np.nanmean(ll.f_phi(sol[:, 0:N]),1)
    ax.plot(time, mean_act, 'k', lw = 1.)
    
    ax.set_xlim([0,150])
    ax.set_ylim([0., 2.1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    a = plt.axes([0.12+0.235*ifact, 0.65, .15, .25])
    
    point1 = -0.5*(frac+1)-0.5*np.sqrt((frac+1)**2-4*(frac)*(1+g_w)+0j)
    point2 = -0.5*(frac+1)+0.5*np.sqrt((frac+1)**2-4*(frac)*(1+g_w)+0j)
    
    
    a.scatter(np.real(w), np.imag(w), 0.05, 0.5*color1+0.5*0.1*np.ones(3))
    
    a1 = np.where((np.abs(np.diff(np.imag(lambda1))))>0.1)
    for i in range(len(a1[0])+1):
        if i == 0 :
            idx_0 = 1
            idx_fin = a1[0][i]-1
        elif i == len(a1[0]):
            idx_fin = len(lambda1)-1
            idx_0 = a1[0][i-1]+1
        else:
            idx_0= a1[0][i-1]+1
            idx_fin = a1[0][i]-1
        a.plot(np.real(lambda1[idx_0:idx_fin]), np.imag(lambda1[idx_0:idx_fin]),  color=[0.2, 0.2, 0.2], lw=0.5)
    a1 = np.where((np.abs(np.diff(np.imag(lambda2))))>0.1)
    for i in range(len(a1[0])+1):
        if i == 0 :
            idx_0 = 1
            idx_fin = a1[0][i]-1
        elif i == len(a1[0]):
            idx_fin = len(lambda2)-1
            idx_0 = a1[0][i-1]+1
        else:
            idx_0= a1[0][i-1]+1
            idx_fin = a1[0][i]-1
        a.plot(np.real(lambda2[idx_0:idx_fin]), np.imag(lambda2[idx_0:idx_fin]),  color=[0.2, 0.2, 0.2], lw=0.5)

    
    cluster1=-frac*(1+g_w)*(xs/(1-xs))+point2
    XL = [-1.0, 0.4]
    a.set_xlim(XL)
    a.set_ylim([-1.7, 1.7])
    a.text(-0.3, 1.4, '$Im(\lambda)$', fontsize=5)
    a.text(0.3, 0.1, '$Re(\lambda)$', fontsize=5)
    
    #a.text('r$Re(\lambda)$')
    
    a.plot([-5,5],[0,0],'k', lw = 0.2)
    a.plot([0,0],[-4,5], 'k', lw = 0.2)
    a.axis('off')
plt.savefig('adap_fig3a.pdf')