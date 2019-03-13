s#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 14:27:57 2018

@author: mbeiran
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import lib_toymodel as ll
from scipy.optimize import fsolve
from scipy import sparse
from scipy.signal import hilbert

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
tau = 1.0
tau_w = 10.0
N = 1000        #Number of neurons
C = 100         #Number of received inputs
g = 4.1        #Inhibition synapse/exc. synapse
f =0.8

JS = [  0.8, 1.0,  1.1, 1.12, 1.15, 1.2, 1.3, 1.4, 1.43, 1.45, 1.5, 1.6, 1.7, 1.8]#0.045,
tau = 1.
T = tau/tau_w
g_w = 0.5
phi_max = 2.0
phi_m = phi_max
gamma = 0.5
#[0.7, 0.9, 1.05, 1.07, 1.1, 1.15, 1.18, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5]# 1.07,  1.1, 1.15, 1.18, 1.2]#
#sigma = 0.50#0.5
counttt = 0
counter = 0
eps = 0.7
Tm = 50
Tm_t = 200
dt = 0.04

#fig = plt.figure(figsize = [28, 15])

iters = 50#25

tau_w = 5. # 5. , 10.
taus = np.logspace(0.1, 1.5,12)
JJJs = [1.1, 1.3,]# 1.0, 1.1, ]
#JJs = [0.8, 1.0, 1.1, 1.3, 1.4, 1.5]
fig1 = plt.figure()

ax1 = fig1.add_subplot(111)

sigma = 0.0
colores = np.zeros((3, len(taus)))
c0 = np.array((0.9, 0.1, 0))
c1 = np.array((0.2, 0.2, 0.2))
c1 = 0.25*c0+0.75*c1

tauselec = np.array((1,11,))
tauselec2 = np.array((1,11))
t_peak2 = np.zeros((len(taus), len(JJJs)))
for i in range(len(taus)):
    colores[:,i] = c0*(len(taus)-i)/(len(taus))+c1*(i)/len(taus)
for iJs, JJs in enumerate(JJJs):
    J = JJs/(np.sqrt(C*f+g**2*C*(1-f)))
    Jef = J*(C*f-g*C*(1-f))
    Jvar = J**2*(C*f+g**2*C*(1-f))

    t_peak = np.zeros(len(taus))
    
    
    for it, tau_w in enumerate(taus):
        Sig = sigma*1.0
        mu_see = np.zeros(iters)
        phi_see = np.zeros( iters)
        spect_see = np.zeros( iters)
        sigma = 0.0
        Sigs= [sigma, ]
        print('tau_w = '+str(tau_w))
        if 1>0:
        #try:
            
            fl = np.load("/Users/mbeiran/Documents/Adaptation/Data/newtau_adaptNoise_tau_w"+str(tau_w)+"_g_w_"+str(g_w)+"_sig_"+str(Sig)+"_J_"+str(JJs)+".npz")
            Cx_all = fl['name1']
            hCx = hilbert(Cx_all)
            Cx_all2 = fl['name1']
            JS  = fl['name2']
            Tt_t = fl['name3']
            mu_save = fl['name4']
            phi_save = fl['name5']
            Del0_save = fl['name6']
            if np.sum(it==tauselec)>0 and iJs == 1:#np.mod(it+1,2)==0:#
                ax1.plot(Tt_t-np.mean(Tt_t), Cx_all, label=r'$\tau_w = $'+str(tau_w)[0:3], color=colores[:, it])
            analytic_signal = hilbert(Cx_all)
            amplitude_envelope = np.abs(analytic_signal)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = (np.diff(instantaneous_phase) /(2.0*np.pi) * (1/dt))
            if np.sum(it==tauselec)>0 and iJs == 1:
                ax1.plot(Tt_t-np.mean(Tt_t), amplitude_envelope, '--', color=colores[:, it], linewidth=0.5)
            namplitude_envelope = amplitude_envelope/np.max(amplitude_envelope)
                
            
            idcs =  np.argsort(Cx_all)
            t_peak[it] = dt*np.abs(idcs[1]-idcs[0])
            fakeCx_all = Cx_all2
            fakeCx_all[min(idcs[1], idcs[0]): max(idcs[1], idcs[0])] = 0.0
            idcs1 = np.argsort(np.abs(namplitude_envelope-(1/np.sqrt(np.e))))
            t_peak2[it, iJs] = dt*np.abs(idcs1[1]-idcs1[0]) 
            if it==0:
                Cx_ra  = np.zeros((len(Cx_all), len(taus)))
            Cx_ra[:, it] = Cx_all
            if np.sum(it==tauselec2)>0 and iJs==1:
                iters = 10
                for ite in range(iters):
                    fl2 = np.load("/Users/mbeiran/Documents/Adaptation/Data/simnewtau_adaptNoise_tau_w"+str(tau_w)+"_g_w_"+str(g_w)+"_sig_"+str(ite)+"_J_"+str(JJs)+".npz")
                    Cx_allsim = fl2['name1']
                    JSsim  = fl2['name2']
                    Tt_tsim = fl2['name3']
                    if ite==0:
                        Cx_Allsim = Cx_allsim/iters
                    else:
                        Cx_Allsim += Cx_allsim/iters
                intv = 40
                if iJs == 1:
                    ax1.scatter(Tt_tsim[::intv]-np.mean(Tt_tsim), np.pi*(Cx_Allsim[::intv]), 10,  edgecolors=colores[:, it], facecolors='none')
                

            
        if 1<0:
        #except:
            J = JJs/(np.sqrt(C*f+g**2*C*(1-f)))#3.7    #J0=0.045 with the parameters
            Jef = J*(C*f-g*C*(1-f))
            
            sseedd = 10
            np.random.seed(sseedd)
            Jvar = J**2*(C*f+g**2*C*(1-f))
    
            Tm = 50
            Tm_t = 200
            dt = 0.04
            Tt = np.arange(0,Tm, dt)
            Tt_t = np.arange(0,Tm_t, dt)
            #dtT  = Tt[1]-Tt[0]
            #SS = 0
            #st = 1
            
            #param=1.
            subtrials = 200
            
            mu_old =-0.1
            phi_old = 0
            for ite in range(iters):
                if np.mod(ite,5)==0:
                    print(ite)
                if ite==0:
                    Sx = J*len(Tt_t)*np.ones(len(Tt_t))/np.sqrt(dt)
                    Snext = Sx
                else:
                    Sx = Snext
            
                mu_new = 0;
                phi_new = 0;
                Del0_new = 0
                for tr in range(subtrials):
                    noise_w = np.sqrt(2)*np.sqrt(Sx)*np.exp(1j*2*np.pi*np.random.rand(len(Sx)))#*np.sqrt(Sx)
                    a1 = np.fft.fftshift(np.fft.ifft((noise_w)))
                    noise_t= np.real(a1)*np.sqrt(len(Tt_t))/np.sqrt(dt)
                    phi_m = phi_max
                    
                    if tr==0:
                        x = np.zeros((len(Tt_t),1))
                        w = np.zeros((len(Tt_t),1))
                        
                        
                        x[0] = np.random.randn(1)
                        w[0] = np.random.randn(1)
                        for i, t in enumerate(Tt_t[:200]):
                            x[i+1] = x[i] + dt*(-x[i]/tau + mu_old/tau+noise_t[i]-g_w*w[i]/tau)+np.sqrt(dt)*np.random.randn()*sigma
                            w[i+1] = w[i] + dt*(-w[i]/tau_w + (x[i]+gamma)/tau_w)
                            
                        x[0] = x[i+1]
                        w[0] = w[i+1]
                            
                    else:
                       x[0] = x[i+1]
                       w[0] = w[i+1]
                       
                    for i, t in enumerate(Tt_t[:-1]):
                            #x[i+1] = x[i] + dt*(-x[i]/tau+mu_old/tau+noise_t[i])
                            x[i+1] = x[i] + dt*(-x[i]/tau + mu_old/tau+noise_t[i]-g_w*w[i]/tau)+np.sqrt(dt)*np.random.randn()*sigma
                            w[i+1] = w[i] + dt*(-w[i]/tau_w + (x[i]+gamma)/tau_w)
                            
                    X = ll.f_phi(x, phi_max= phi_m)
                    
                    mu_new +=  (Jef *np.mean(X))/subtrials#(Jef *np.mean(X)-g_w*np.mean(x))/subtrials
                    phi_new += np.mean(X)/subtrials
                    Del0_new += np.var(x)/subtrials
                    
                    #second tour
                    w2, PSD2 = ll.PSDfu(X-phi_old, Tt_t)
                    PSD2 = PSD2[:,0]
                    
                    if tr==0:
                        S2 = PSD2/subtrials
                    else:
                        S2 += PSD2/subtrials
    
                    if ite ==iters-1:
                        ww2, PSD_x = ll.PSDfu(x, Tt_t)
                        PSD_x = PSD_x[:,0]
                        if tr==0:
                            S_X = PSD_x/subtrials
                        else:
                            S_X += PSD_x/subtrials 
                
                spect_see[ ite] = np.mean(np.abs(S2-Sx))
                if ite>10:
                    fact = 0.6
                else:
                    fact = 1.0
                Snext = (1-eps*fact)*Snext+(eps*fact)*Jvar*S2+(eps*fact)*((2./Tm_t)*sigma**2)
                
                mu_see[ ite] = mu_old
                phi_see[ ite] = phi_old
                phi_old = phi_new
                mu_old = (1-eps*fact)*mu_old+(eps*fact)*mu_new
                
            mu_save = mu_old;
            phi_save = phi_old;
            Del0_save = Del0_new
    
            b = np.fft.fftshift(np.fft.ifft(Snext))
            Cx = np.real(b)/dt
            
            Cx_all=Cx
            
    
            np.savez("newtau_adaptNoise_tau_w"+str(tau_w)+"_g_w_"+str(g_w)+"_sig_"+str(Sig)+"_J_"+str(JJs), name1 = Cx_all, name2 = JS, name3 = Tt_t, name4 = mu_save, name5 = phi_save, name6 = Del0_save )
plt.legend( loc=1)
ax1.set_xlim([-30, 55])
ax1.set_xlabel('time lag')
ax1.set_ylabel('rate autocorrelation')

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')
#plt.yticks([0, 0.5, 1])
plt.savefig('adap_fig4_A.pdf')



#%%
FL = np.load("/Users/mbeiran/Documents/Adaptation/syn_timescales.npz")#, name1 = t_peak2, name2 = taus)
tpeak_syn = FL['name1']
taus_syn  = FL['name2']
FL = np.load("/Users/mbeiran/Documents/Adaptation/syn_timescales_J_"+str(1.1)+ ".npz")#, name1 = t_peak2, name2 = taus)
tpeak_syn2 = FL['name1']
taus_syn2  = FL['name2']


#%%
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(taus, t_peak2[:,0], '-', color=[0.7, 0.4 , 0.3] ,  lw=2, label=r'$J_{cs}=1.1$')
ax.plot(taus, t_peak2[:,1], '-', color=[0.8, 0. , 0.] ,  lw=2, label=r'$J_{cs}=1.3$')

ax.legend()
ax.set_xlabel(r'time constant $\tau_w$')
ax.set_ylabel('correlation time')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.savefig('adap_fig4_C.pdf')




#%%
FL = np.load("/Users/mbeiran/Documents/Adaptation/syn_timescales.npz")#, name1 = t_peak2, name2 = taus)
tpeak_syn = FL['name1']
taus_syn  = FL['name2']
fig = plt.figure()
ax2 = fig.add_subplot(111)
ax2.tick_params('y', colors=[0.3, 0.3, 0.3])
ax2.plot(taus_syn2, tpeak_syn2[0], '-', color=[0.7, 0.7 , 0.7] ,  lw=2, label='$J_{cs}=1.1$')
ax2.plot(taus_syn, tpeak_syn, '-', color=[0.3, 0.3 , 0.3] ,  lw=2, label='$J_{cs}=1.3$')
ax2.set_xlabel(r'time constant $\tau_s$')
ax2.set_ylabel('correlation time')
ax2.set_ylim([0,300])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.yaxis.set_ticks_position('left')
ax2.xaxis.set_ticks_position('bottom')
ax2.legend()
plt.savefig('adap_fig4_D.pdf')

#%%
#%%
tau = 1.0
tau_w = 10.0
N = 1000        #Number of neurons
C = 100         #Number of received inputs
g = 4.1        #Inhibition synapse/exc. synapse
f =0.8

JS = [  0.8, 1.0,  1.1, 1.12, 1.15, 1.2, 1.3, 1.4, 1.43, 1.45, 1.5, 1.6, 1.7, 1.8]#0.045,
tau = 1.
T = tau/tau_w
g_w = 0.5
phi_max = 2.0
phi_m = phi_max
gamma = 0.5
#[0.7, 0.9, 1.05, 1.07, 1.1, 1.15, 1.18, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5]# 1.07,  1.1, 1.15, 1.18, 1.2]#
#sigma = 0.50#0.5
counttt = 0
counter = 0
eps = 0.7
Tm = 50
Tm_t = 200
dt = 0.04

#fig = plt.figure(figsize = [28, 15])

iters = 50#25

tau_w = 5. # 5. , 10.
taus = np.logspace(0.1, 1.5,12)
JJJs = [1.1,]# 1.0, 1.1, ]
#JJs = [0.8, 1.0, 1.1, 1.3, 1.4, 1.5]
fig1 = plt.figure()

ax1 = fig1.add_subplot(111)

sigma = 0.0
colores = np.zeros((3, len(taus)))
c0 = np.array((0.9, 0.1, 0))
c1 = np.array((0.2, 0.2, 0.2))
c1 = 0.25*c0+0.75*c1

tauselec = np.array((1,11,))
tauselec2 = np.array((1,11))
t_peak2 = np.zeros((len(taus), len(JJJs)))
colores2 = np.zeros((3,2))
colores2[:,0] = [0.42, 0.4 , 0.48];
colores2[:,1] = [0.13, 0.14 , 0.15];
selec= 0
for i in range(len(taus)):
    colores[:,i] = c0*(len(taus)-i)/(len(taus))+c1*(i)/len(taus)
for iJs, JJs in enumerate(JJJs):
    J = JJs/(np.sqrt(C*f+g**2*C*(1-f)))
    Jef = J*(C*f-g*C*(1-f))
    Jvar = J**2*(C*f+g**2*C*(1-f))

    t_peak = np.zeros(len(taus))
    
    
    for it, tau_w in enumerate(taus):
        Sig = sigma*1.0
        mu_see = np.zeros(iters)
        phi_see = np.zeros( iters)
        spect_see = np.zeros( iters)
        sigma = 0.0
        Sigs= [sigma, ]
        print('tau_w = '+str(tau_w))
        if 1>0:
        #try:
            
            fl = np.load("/Users/mbeiran/Documents/Adaptation/synnewtau_adaptNoise_tau_w"+str(tau_w)+"_g_w_"+str(g_w)+"_sig_"+str(3)+"_J_"+str(JJs)+".npz")
            Cx_all = fl['name1']
            hCx = hilbert(Cx_all)
            Cx_all2 = fl['name1']
            JS  = fl['name2']
            Tt_t = fl['name3']
            mu_save = fl['name4']
            phi_save = fl['name5']
            Del0_save = fl['name6']
            intv = 35
            if np.sum(it==tauselec)>0:#np.mod(it+1,2)==0:#
                ax1.plot(Tt_t[::intv]-np.mean(Tt_t[::intv]), Cx_all[::intv], label=r'$\tau_w = $'+str(tau_w)[0:3], color=colores2[:, selec])
            analytic_signal = hilbert(Cx_all)
            amplitude_envelope = np.abs(analytic_signal)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = (np.diff(instantaneous_phase) /(2.0*np.pi) * (1/dt))
            if np.sum(it==tauselec)>0 and iJs == 1:
                ax1.plot(Tt_t-np.mean(Tt_t), amplitude_envelope, '--', color=colores2[:, selec], linewidth=0.5)
                
            namplitude_envelope = amplitude_envelope/np.max(amplitude_envelope)
                
            
            idcs =  np.argsort(Cx_all)
            t_peak[it] = dt*np.abs(idcs[1]-idcs[0])
            fakeCx_all = Cx_all2
            fakeCx_all[min(idcs[1], idcs[0]): max(idcs[1], idcs[0])] = 0.0
            idcs1 = np.argsort(np.abs(namplitude_envelope-(1/np.sqrt(np.e))))
            t_peak2[it, iJs] = dt*np.abs(idcs1[1]-idcs1[0]) 
            if it==0:
                Cx_ra  = np.zeros((len(Cx_all), len(taus)))
            Cx_ra[:, it] = Cx_all
            if np.sum(it==tauselec2)>0:
                iters = 10
                for ite in range(iters):
                    fl2 = np.load("/Users/mbeiran/Documents/Adaptation/Data/simsynnewtau_adaptNoise_tau_w"+str(tau_w)+"_g_w_"+str(g_w)+"_sig_"+str(2)+"_J_"+str(JJs)+".npz")
                    Cx_allsim = fl2['name1']
                    JSsim  = fl2['name2']
                    Tt_tsim = fl2['name3']
                    if ite==0:
                        Cx_Allsim = Cx_allsim/iters
                    else:
                        Cx_Allsim += Cx_allsim/iters

                ax1.scatter(Tt_tsim[::intv]-np.mean(Tt_tsim[::intv]), np.pi*Cx_Allsim[::intv], 10,  edgecolors=colores2[:, selec], facecolors='none')
                selec+=1

            
        if 1<0:
        #except:
            J = JJs/(np.sqrt(C*f+g**2*C*(1-f)))#3.7    #J0=0.045 with the parameters
            Jef = J*(C*f-g*C*(1-f))
            
            sseedd = 10
            np.random.seed(sseedd)
            Jvar = J**2*(C*f+g**2*C*(1-f))
    
            Tm = 50
            Tm_t = 200
            dt = 0.04
            Tt = np.arange(0,Tm, dt)
            Tt_t = np.arange(0,Tm_t, dt)
            #dtT  = Tt[1]-Tt[0]
            #SS = 0
            #st = 1
            
            #param=1.
            subtrials = 200
            
            mu_old =-0.1
            phi_old = 0
            for ite in range(iters):
                if np.mod(ite,5)==0:
                    print(ite)
                if ite==0:
                    Sx = J*len(Tt_t)*np.ones(len(Tt_t))/np.sqrt(dt)
                    Snext = Sx
                else:
                    Sx = Snext
            
                mu_new = 0;
                phi_new = 0;
                Del0_new = 0
                for tr in range(subtrials):
                    noise_w = np.sqrt(2)*np.sqrt(Sx)*np.exp(1j*2*np.pi*np.random.rand(len(Sx)))#*np.sqrt(Sx)
                    a1 = np.fft.fftshift(np.fft.ifft((noise_w)))
                    noise_t= np.real(a1)*np.sqrt(len(Tt_t))/np.sqrt(dt)
                    phi_m = phi_max
                    
                    if tr==0:
                        x = np.zeros((len(Tt_t),1))
                        w = np.zeros((len(Tt_t),1))
                        
                        
                        x[0] = np.random.randn(1)
                        w[0] = np.random.randn(1)
                        for i, t in enumerate(Tt_t[:200]):
                            x[i+1] = x[i] + dt*(-x[i]/tau + mu_old/tau+noise_t[i]-g_w*w[i]/tau)+np.sqrt(dt)*np.random.randn()*sigma
                            w[i+1] = w[i] + dt*(-w[i]/tau_w + (x[i]+gamma)/tau_w)
                            
                        x[0] = x[i+1]
                        w[0] = w[i+1]
                            
                    else:
                       x[0] = x[i+1]
                       w[0] = w[i+1]
                       
                    for i, t in enumerate(Tt_t[:-1]):
                            #x[i+1] = x[i] + dt*(-x[i]/tau+mu_old/tau+noise_t[i])
                            x[i+1] = x[i] + dt*(-x[i]/tau + mu_old/tau+noise_t[i]-g_w*w[i]/tau)+np.sqrt(dt)*np.random.randn()*sigma
                            w[i+1] = w[i] + dt*(-w[i]/tau_w + (x[i]+gamma)/tau_w)
                            
                    X = ll.f_phi(x, phi_max= phi_m)
                    
                    mu_new +=  (Jef *np.mean(X))/subtrials#(Jef *np.mean(X)-g_w*np.mean(x))/subtrials
                    phi_new += np.mean(X)/subtrials
                    Del0_new += np.var(x)/subtrials
                    
                    #second tour
                    w2, PSD2 = ll.PSDfu(X-phi_old, Tt_t)
                    PSD2 = PSD2[:,0]
                    
                    if tr==0:
                        S2 = PSD2/subtrials
                    else:
                        S2 += PSD2/subtrials
    
                    if ite ==iters-1:
                        ww2, PSD_x = ll.PSDfu(x, Tt_t)
                        PSD_x = PSD_x[:,0]
                        if tr==0:
                            S_X = PSD_x/subtrials
                        else:
                            S_X += PSD_x/subtrials 
                
                spect_see[ ite] = np.mean(np.abs(S2-Sx))
                if ite>10:
                    fact = 0.6
                else:
                    fact = 1.0
                Snext = (1-eps*fact)*Snext+(eps*fact)*Jvar*S2+(eps*fact)*((2./Tm_t)*sigma**2)
                
                mu_see[ ite] = mu_old
                phi_see[ ite] = phi_old
                phi_old = phi_new
                mu_old = (1-eps*fact)*mu_old+(eps*fact)*mu_new
                
            mu_save = mu_old;
            phi_save = phi_old;
            Del0_save = Del0_new
    
            b = np.fft.fftshift(np.fft.ifft(Snext))
            Cx = np.real(b)/dt
            
            Cx_all=Cx
            
    
            np.savez("newtau_adaptNoise_tau_w"+str(tau_w)+"_g_w_"+str(g_w)+"_sig_"+str(Sig)+"_J_"+str(JJs), name1 = Cx_all, name2 = JS, name3 = Tt_t, name4 = mu_save, name5 = phi_save, name6 = Del0_save )
plt.legend( loc=1)
ax1.set_xlim([-30, 55])
#ax1.set_ylim([-1, 1])

ax1.set_xlabel('time lag')
ax1.set_ylabel('rate autocorrelation')

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')
#plt.yticks([0, 0.5, 1])
plt.savefig('adap_fig4_B.pdf')
