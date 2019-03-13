# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 16:19:03 2016

@author: mbeiran
"""

import numpy as np
from scipy import sparse
from matplotlib.ticker import MultipleLocator
from scipy.integrate import odeint
from math import erfc, erf
from scipy.integrate import quad
#import sympy.mpmath as mp
import mpmath as mp

def eig_limitcurve(x, frac, g_w):
    trace = (frac+1-x+0j)**2-4*frac*(1+g_w-x+0j)
    fix = -frac-1+x
    lambda1 = 0.5*(fix+np.sqrt(trace))
    lambda2 = 0.5*(fix-np.sqrt(trace))
    return(lambda1, lambda2)
    
def eig_limitcurve_aproxright(x, frac, g_w):
    lambda1 = -0.5*frac*(1+((1-x+2*g_w)/(1-x)))
    lambda2 = -1+x+frac*g_w*(1-np.conj(x))/(np.abs(1-x))**2
    return(lambda1, lambda2)
    
def f_phi(x, phi_max=1000000, gamma=0.5, ):    

    '''
    Threshold-linear activation function with upper bound
    
    INPUT
    -----
    x: input domain
    phi_max: upper bound, default infinity
    gamma: offset, default 0.5
f    
    OUTPUT
    ------
    res: transformed values of the input array x
    '''
    
    res = np.clip(x,-gamma, phi_max-gamma)+gamma
    return(res)

def f_phi_sym(x, phi_max=1000000):    

    '''
    Threshold-linear activation function with upper bound
    
    INPUT
    -----
    x: input domain
    phi_max: upper bound, default infinity
    gamma: offset, default 0.5
    
    OUTPUT
    ------
    res: transformed values of the input array x
    '''
    
    res = np.clip(x,-phi_max, phi_max)
    return(res)
    
def F_phi(x, phi_max=1000000, gamma=0.5, ):    

    '''
    Threshold-linear activation function with upper bound
    
    INPUT
    -----
    x: input domain
    phi_max: upper bound, default infinity
    gamma: offset, default 0.5
    
    OUTPUT
    ------
    res: transformed values of the input array x
    '''
    if np.sum(x)!=np.min(x):
        res = np.zeros(len(x))
        m1 = [(x>-gamma)*(x<phi_max)]    
        res[m1] = (x[m1]+gamma)**2/(2.0)
        m2 = x>(phi_max-gamma)
        res[m2] = phi_max*(gamma+x[m2]-0.5*phi_max)
    else:
        if x<-gamma:
            res=0
        elif x<-phi_max:
            res=(x+gamma)**2/(2.0)
        else:
            res=phi_max*(gamma+x-0.5*phi_max)
    return(res)


def phi_der(x, phi_max=1000000, gamma=0.5, ):    

    '''
    Threshold-linear activation function with upper bound
    
    INPUT
    -----
    x: input domain
    phi_max: upper bound, default infinity
    gamma: offset, default 0.5
    
    OUTPUT
    ------
    res: transformed values of the input array x
    '''
    
    res = np.ones(np.shape(x))

    mask_0 = (x<-gamma)
    mask_bound = (x>(phi_max-gamma))
    
    res[mask_0] = 0.
    res[mask_bound] = 0.
    
    return(res)
    
def create_Jmat(N, C, g=4.5, f=0.8, seed=69):    

    '''
    Create random connectivity matrix (divided by coupling constant J)
    
    INPUT
    -----
    N: number of neurons
    C: number of connections
    f: proportion excitatory vs inhibitory neurons
    
    OUTPUT
    ------
    Jmat: NxN connectivity matrix. Needs to be multiplied by J
    '''
    
    #Choose inhibitory and excitatory cells  
    np.random.seed(seed)
    perms = np.random.permutation(N)
    mask_exc = perms[0:int(f*N)]
    mask_inh = perms[int(f*N):]
    Jmat = np.zeros((N,N))
    
    #Determine connections
    for neur in range(N):
        Jmat[neur,mask_exc[np.random.permutation(int(f*N))[0:int(f*C)]]]=1.0
        Jmat[neur,mask_inh[np.random.permutation(int((1-f)*N))[0:(C-int(f*C))]]]=-g

    
    return(Jmat)
    
def rate_simulation(T, init_cond, Jmat, f_phi, phi_max=1e7, gamma=0.5, dt = 0.05):
    N = len(init_cond)
    time = np.arange(0, T,dt)
    x_t = np.zeros((len(time), N))
    x_t[0,:]=init_cond
    for i in range(len(time)-1):
        x_t[i+1,:]=x_t[i,:]+dt*(-x_t[i,:]+np.dot(Jmat, f_phi(x_t[i,:], 
                                                phi_max=phi_max, gamma=gamma)))
    
    return(x_t, time)
    

def rate_eqs(xws, time, Q1, Q2,  phimax):
    ph = f_phi_sym(xws, phi_max=phimax)
    dxws = Q1.dot(xws)+Q2.dot(ph)
    return(dxws)

    
def func_bisect(r, frac, g_w):
    ts = np.arange(0.00001,2*np.pi, 0.001)
    xs_m = r*np.exp(1j*ts)
    lambda1_m, lambda2_m = eig_limitcurve(xs_m, frac, g_w)
    sym_m = np.max(np.max(np.real(lambda1_m)), np.max(np.real(lambda2_m)))        
    return(sym_m)

def rate_simfinal(time, xws0, Jmat, tau, tau_w, g_w, phimax, method='rk4', numbs =300):
    
    N = int(len(xws0)/2)
    Q1 = -1.0*np.eye(2*N)
    Q1[0:N, N:] = -g_w*np.eye(N)
    Q1[0:N, :] *= (1./tau)
    Q1[N:, :] *= (1./tau_w)
    
    q1 = sparse.csc_matrix(Q1)
    
    Q2 = np.zeros((2*N, 2*N))
    Q2[0:N, 0:N] = (1./tau)*Jmat
    
    Q3 = np.zeros((2*N, 2*N))
    Q3[N:, 0:N] = (1./tau_w)*np.eye(N)
    
    meanr=0
    meana=0
    
    q2 = sparse.csc_matrix(Q2)
    q3 = sparse.csc_matrix(Q3)
    
    if method=='':    
        sol = odeint(rate_eqs_final, xws0, time, args=(q1, q2, phimax))
        sol2 = 0
    elif method=='rk4':
        sol, sol2, meanr, meana = my_rk4_final(rate_eqs_final, xws0, time, q1, q2, q3, phimax, num = numbs)

    else:
        sol, sol2 = my_euler(rate_eqs_final, xws0, time, q1, q2, phimax)
        meanr=0
        meana=0
    return(sol, sol2, meanr, meana, q1, q2)

def rate_simnoisefinal(time, xws0, Jmat, tau, tau_w, g_w, phimax, sigma, method='rk4', numbs =300):
    
    N = int(len(xws0)/2)
    Q1 = -1.0*np.eye(2*N)
    Q1[0:N, N:] = -g_w*np.eye(N)
    Q1[0:N, :] *= (1./tau)
    Q1[N:, :] *= (1./tau_w)
    
    q1 = sparse.csc_matrix(Q1)
    
    Q2 = np.zeros((2*N, 2*N))
    Q2[0:N, 0:N] = (1./tau)*Jmat
    
    Q3 = np.zeros((2*N, 2*N))
    Q3[N:, 0:N] = (1./tau_w)*np.eye(N)
    
    meanr=0
    meana=0
    
    q2 = sparse.csc_matrix(Q2)
    q3 = sparse.csc_matrix(Q3)
    
    if method=='':    
        sol = odeint(rate_eqs_final, xws0, time, args=(q1, q2, phimax))
        sol2 = 0
    elif method=='rk4':
        sol, sol2, meanr, meana = my_rk4noise_final(rate_eqs_final, xws0, time, q1, q2, q3, phimax, sigma, num = numbs)

    else:
        sol, sol2 = my_euler(rate_eqs_final, xws0, time, q1, q2, phimax)
        meanr=0
        meana=0
    return(sol, sol2, meanr, meana, q1, q2)
    
def rate_eqs_final(xws, time, Q1, Q2, Q3,  phimax, gamma=0.5):
    ph = f_phi_sym(xws, phi_max=phimax)
    dxws = Q1.dot(xws)+Q2.dot(ph)+Q3.dot(xws-gamma)
    return(dxws)
    
def rate_sim(time, xws0, Jmat, tau, tau_w, g_w, phimax, method='', numbs =300):
    
    N = int(len(xws0)/2)
    Q1 = -1.0*np.eye(2*N)
    Q1[0:N, N:] = -g_w*np.eye(N)
    Q1[0:N, :] *= (1./tau)
    Q1[N:, :] *= (1./tau_w)
    
    q1 = sparse.csc_matrix(Q1)
    
    Q2 = np.zeros((2*N, 2*N))
    Q2[0:N, 0:N] = (1./tau)*Jmat
    Q2[N:, 0:N] = (1./tau_w)*np.eye(N)
    
    meanr=0
    meana=0
    
    q2 = sparse.csc_matrix(Q2)
    if method=='':    
        sol = odeint(rate_eqs, xws0, time, args=(q1, q2, phimax))
        sol2 = 0
    elif method=='rk4':
        sol, sol2, meanr, meana = my_rk4(rate_eqs, xws0, time, q1, q2, phimax, num = numbs)

    else:
        sol, sol2 = my_euler(rate_eqs, xws0, time, q1, q2, phimax)
        meanr=0
        meana=0
    return(sol, sol2, meanr, meana, q1, q2)
    
def rate_sim_linadap(time, xws0, Jmat, tau, tau_w, g_w, phimax, method='', numbs =300):
    
    N = int(len(xws0)/2)
    Q1 = -1.0*np.eye(2*N)
    Q1[0:N, N:] = -g_w*np.eye(N)
    Q1[0:N, :] *= (1./tau)
    Q1[N:, 0:N] = np.eye(N)
    Q1[N:, :] *= (1./tau_w)

    q1 = sparse.csc_matrix(Q1)
    
    Q2 = np.zeros((2*N, 2*N))
    Q2[0:N, 0:N] = (1./tau)*Jmat
    
    
    meanr=0
    meana=0
    
    q2 = sparse.csc_matrix(Q2)
    if method=='':    
        sol = odeint(rate_eqs, xws0, time, args=(q1, q2, phimax))
        sol2 = 0
    elif method=='rk4':
        sol, sol2, meanr, meana = my_rk4(rate_eqs, xws0, time, q1, q2, phimax, num = numbs)

    else:
        sol, sol2 = my_euler(rate_eqs, xws0, time, q1, q2, phimax)
        meanr=0
        meana=0
    return(sol, sol2, meanr, meana, q1, q2)
    
def rate_sim_noise(time, xws0, Jmat, tau, tau_w, g_w, sigma, phimax, method='', numbs =300):
    
    N = int(len(xws0)/2)
    Q1 = -1.0*np.eye(2*N)
    Q1[0:N, N:] = -g_w*np.eye(N)
    Q1[0:N, :] *= (1./tau)
    Q1[N:, 0:N] = np.eye(N)
    Q1[N:, :] *= (1./tau_w)
    
    q1 = sparse.csc_matrix(Q1)
    
    Q2 = np.zeros((2*N, 2*N))
    Q2[0:N, 0:N] = (1./tau)*Jmat
    #Q2[N:, 0:N] = (1./tau_w)*np.eye(N)
    
    meanr=0
    meana=0
    
    q2 = sparse.csc_matrix(Q2)
    if method=='':    
        sol = odeint(rate_eqs, xws0, time, args=(q1, q2, phimax))
        sol2 = 0
    elif method=='rk4':
        sol, sol2, meanr, meana = my_rk4_noise(rate_eqs, xws0, time, q1, q2, phimax, sigma, num = numbs)

    else:
        sol, sol2 = my_euler(rate_eqs, xws0, time, q1, q2, phimax)
        meanr=0
        meana=0
    return(sol, sol2, meanr, meana, q1, q2)
    
def rate_spks(time, dt1, xws0, Jmat, tau, tau_w, g_w, phimax, method='', numbs =300):
    dt = time[1]-time[0]
    time1 = np.arange(0, time[-1], dt1)
    N = len(xws0)/2
    Q1 = -1.0*np.eye(2*N)
    Q1[0:N, N:] = -g_w*np.eye(N)
    Q1[0:N, :] *= (1./tau)
    Q1[N:, :] *= (1./tau_w)

    q1 = sparse.csc_matrix(Q1)
    
    Q2 = np.zeros((2*N, 2*N))
    #Q2[0:N, 0:N] = (1./tau)*Jmat
    Q2[N:, 0:N] = (1./tau_w)*np.eye(N)
    
    Q3 = np.zeros((2*N, 2*N))
    Q3[0:N, 0:N] = (1./tau)*Jmat
    
        
    q2 = sparse.csc_matrix(Q2)
    q3 = sparse.csc_matrix(Q3)
    rates = np.zeros((len(time), 2*N))

    rates_last = np.zeros(((len(time)/50),len(xws0)))
    counter = 0
    for i, ti in enumerate(time1):
        if i==0:
            rates_old=xws0
            rr = np.random.rand(N)
            spks_old = (1./dt)*(rr<(f_phi(rates_old[0:N], phi_max=phimax)*dt))
            spks_old = np.concatenate((spks_old, np.zeros(N)))
        else:
            rates_old =rates_old+dt*new_rate_eqs(rates_old, spks_old, time, q1, q2, q3, phimax)
            rr = np.random.rand(N)
            spks_old=(1./dt)*(rr<(f_phi(rates_old[0:N], phi_max=phimax)*dt))
            spks_old = np.concatenate((spks_old, np.zeros(N)))
        if ti == time[counter]:
            rates[counter,:]=rates_old          
            counter +=1
            
        if counter>49*len(time)/50:
            if counter == 1 + (49*len(time)/50):
                I = counter
            rates_last[counter-I,:]=rates_old
    sol = rates
    sol2 = 0
    mean_r=np.mean(f_phi(rates_old[0:N],phi_max=phimax))
    mean_a=np.mean(rates_old[N:])
    return(sol, sol2, rates_last, mean_r, mean_a)

def rate_simfinal_adap(time, xws0, Jmat, tau, tau_w, g_w, phimax, method='rk4', numbs =300):
    
    N = int(len(xws0)/2)
    Q1 = -1.0*np.eye(2*N)
    Q1[0:N, N:] = np.eye(N)
    Q1[0:N, :] *= (1./tau)
    Q1[N:, :] *= (1./tau_w)
#    
#                        Q = np.zeros((2*N,2*N))
#                    Q[0:N,0:N]=-np.eye(N)
#                    Q[0:N,N:]=np.eye(N)
#                    Q[N:,0:N]=-Jmat*frac
#                    Q[N:,N:]=-frac*np.eye(N)
    q1 = sparse.csc_matrix(Q1)
    
    Q2 = np.zeros((2*N, 2*N))
    Q2[N:, 0:N] = (1./tau_w)*Jmat
    

    
    meanr=0
    meana=0
    
    q2 = sparse.csc_matrix(Q2)

    
    if method=='':    
        sol = odeint(rate_eqs_final, xws0, time, args=(q1, q2, phimax))
        sol2 = 0
    elif method=='rk4':
        sol, sol2, meanr, meana = my_rk4(rate_eqs, xws0, time, q1, q2,  phimax, num = numbs)

    else:
        sol, sol2 = my_euler(rate_eqs, xws0, time, q1, q2, phimax)
        meanr=0
        meana=0
    return(sol, sol2, meanr, meana, q1, q2)
    

def rate_simfinalnoise_adap(time, xws0, Jmat, tau, tau_w, g_w, phimax, sigma, method='rk4', numbs =300):
    
    N = int(len(xws0)/2)
    Q1 = -1.0*np.eye(2*N)
    Q1[0:N, N:] = np.eye(N)
    Q1[0:N, :] *= (1./tau)
    Q1[N:, :] *= (1./tau_w)
#    
#                        Q = np.zeros((2*N,2*N))
#                    Q[0:N,0:N]=-np.eye(N)
#                    Q[0:N,N:]=np.eye(N)
#                    Q[N:,0:N]=-Jmat*frac
#                    Q[N:,N:]=-frac*np.eye(N)
    q1 = sparse.csc_matrix(Q1)
    
    Q2 = np.zeros((2*N, 2*N))
    Q2[N:, 0:N] = (1./tau_w)*Jmat
    

    
    meanr=0
    meana=0
    
    q2 = sparse.csc_matrix(Q2)

    
    if method=='':    
        sol = odeint(rate_eqs_final, xws0, time, args=(q1, q2, phimax))
        sol2 = 0
    elif method=='rk4':
        sol, sol2, meanr, meana = my_rk4_noise(rate_eqs, xws0, time, q1, q2,  phimax, sigma, num = numbs)

    else:
        sol, sol2 = my_euler(rate_eqs, xws0, time, q1, q2, sigma, phimax)
        meanr=0
        meana=0
    return(sol, sol2, meanr, meana, q1, q2)
    
    
def new_rate_eqs(rates, spks, time, Q1, Q2, Q3, phimax):
    ph = f_phi(rates, phi_max=phimax)
    dxws = Q1.dot(rates)+Q2.dot(ph)+Q3.dot(spks)
    return(dxws)
    

def PSD(signal, t):
    dt = (t[1]-t[0])
    T = t[-1]
    FTsignal = np.fft.fft(signal,axis = 0)
    FTfreq = np.fft.fftfreq(len(signal),d=dt)
    PSD = np.real((FTsignal*np.conj(FTsignal)))*(dt*dt/T)
    mask = FTfreq>0
    return FTfreq[mask], PSD[mask]
def PSDfu(signal, t):
    dt = (t[1]-t[0])
    T = t[-1]
    FTsignal = np.fft.fft(signal,axis = 0)
    FTfreq = np.fft.fftfreq(len(signal),d=dt)
    PSD = np.real((FTsignal*np.conj(FTsignal)))*(dt*dt/T)
    
    return FTfreq, PSD

def my_euler(f, x0, time, q1, q2, phimax):
    rates = np.zeros((len(time), len(x0)))
    dt = time[1]-time[0]
    for i, ti in enumerate(time):
        if i==0:
            rates[0,:]=x0
        else:
            rates[i,:]=rates[i-1,:]+dt*rate_eqs(rates[i-1,:], time, q1, q2,  phimax)
    return rates, 0


def my_rk4(f, x0, time, q1, q2, phimax, num=300):
    rates = np.zeros((len(time), num))
    rates_old = x0
    dt = time[1]-time[0]
    N = int(len(x0)/2)
    rates_last = np.zeros(((int(len(time)/50)),len(x0)))
    meanr= np.zeros(len(time))
    meana = np.zeros(len(time))
    for i, ti in enumerate(time):
        if i==0:
            rates[i,0:int(num/2)]=x0[0:int(num/2)]
            rates[i,int(num/2):]=x0[N:N+int(num/2)]
        else:
            k1 = rate_eqs(rates_old, dt, q1, q2,  phimax)
            k2 = rate_eqs(rates_old+(dt/2.)*k1, dt, q1, q2,  phimax)
            k3 = rate_eqs(rates_old+(dt/2.)*k2, dt, q1, q2,  phimax)
            k4 = rate_eqs(rates_old+(dt)*k3, dt, q1, q2,  phimax)
            rates_old=rates_old+(dt/6.)*(k1+2*k2+2*k3+k4)
            rates[i,0:int(num/2)]=rates_old[0:int(num/2)]
            rates[i,int(num/2):]=rates_old[N:N+int(num/2)]
        if i>49*len(time)/50:
            if i == 1 + int(49*len(time)/50):
                I = i
            rates_last[i-I,:]=rates_old
        meanr[i]=0#np.mean(f_phi_sim(rates_old[0:N],phi_max=phimax))
        meana[i]=0#np.mean(rates_old[N:])
    return rates, rates_last, meanr, meana

def my_rk4_final(f, x0, time, q1, q2, q3, phimax, num=300):
    rates = np.zeros((len(time), num))
    rates_old = x0
    dt = time[1]-time[0]
    N = int(len(x0)/2)
    rates_last = np.zeros(((int(len(time)/50)),len(x0)))
    meanr= np.zeros(len(time))
    meana = np.zeros(len(time))
    for i, ti in enumerate(time):
        if i==0:
            rates[i,0:int(num/2)]=x0[0:int(num/2)]
            rates[i,int(num/2):]=x0[N:N+int(num/2)]
        else:
            k1 = rate_eqs_final(rates_old, dt, q1, q2, q3, phimax)
            k2 = rate_eqs_final(rates_old+(dt/2.)*k1, dt, q1, q2, q3,  phimax)
            k3 = rate_eqs_final(rates_old+(dt/2.)*k2, dt, q1, q2,  q3, phimax)
            k4 = rate_eqs_final(rates_old+(dt)*k3, dt, q1, q2, q3, phimax)
            rates_old=rates_old+(dt/6.)*(k1+2*k2+2*k3+k4)
            rates[i,0:int(num/2)]=rates_old[0:int(num/2)]
            rates[i,int(num/2):]=rates_old[N:N+int(num/2)]
        if i>49*len(time)/50:
            if i == 1 + int(49*len(time)/50):
                I = i
            rates_last[i-I,:]=rates_old
        meanr[i]=0#np.mean(f_phi_sim(rates_old[0:N],phi_max=phimax))
        meana[i]=0#np.mean(rates_old[N:])
    return rates, rates_last, meanr, meana

def my_rk4noise_final(f, x0, time, q1, q2, q3, phimax, sigma, num=300):
    rates = np.zeros((len(time), num))
    rates_old = x0
    dt = time[1]-time[0]
    sqdt = np.sqrt(dt)
    N = int(len(x0)/2)
    rates_last = np.zeros(((int(len(time)/50)),len(x0)))
    meanr= np.zeros(len(time))
    meana = np.zeros(len(time))
    for i, ti in enumerate(time):
        if i==0:
            rates[i,0:int(num/2)]=x0[0:int(num/2)]
            rates[i,int(num/2):]=x0[N:N+int(num/2)]
        else:
            k1 = rate_eqs_final(rates_old, dt, q1, q2, q3, phimax)
            k2 = rate_eqs_final(rates_old+(dt/2.)*k1, dt, q1, q2, q3,  phimax)
            k3 = rate_eqs_final(rates_old+(dt/2.)*k2, dt, q1, q2,  q3, phimax)
            k4 = rate_eqs_final(rates_old+(dt)*k3, dt, q1, q2, q3, phimax)
            noise_vec = np.hstack((np.random.randn(N), np.zeros(N)))
            rates_old=rates_old+(dt/6.)*(k1+2*k2+2*k3+k4)+sqdt*sigma*noise_vec
            rates[i,0:int(num/2)]=rates_old[0:int(num/2)]
            rates[i,int(num/2):]=rates_old[N:N+int(num/2)]
        if i>49*len(time)/50:
            if i == 1 + int(49*len(time)/50):
                I = i
            rates_last[i-I,:]=rates_old
        meanr[i]=0#np.mean(f_phi_sim(rates_old[0:N],phi_max=phimax))
        meana[i]=0#np.mean(rates_old[N:])
    return rates, rates_last, meanr, meana

def my_rk4_noise(f, x0, time, q1, q2, phimax, sigma, num=300):
    rates = np.zeros((len(time), num))
    rates_old = x0
    dt = time[1]-time[0]
    N = int(len(x0)/2)
    rates_last = np.zeros(((int(len(time)/50)),len(x0)))
    meanr= np.zeros(len(time))
    meana = np.zeros(len(time))
    for i, ti in enumerate(time):
        if i==0:
            rates[i,0:int(num/2)]=x0[0:int(num/2)]
            rates[i,int(num/2):]=x0[N:N+int(num/2)]
        else:
            k1 = rate_eqs(rates_old, dt, q1, q2,  phimax)
            k2 = rate_eqs(rates_old+(dt/2.)*k1, dt, q1, q2,  phimax)
            k3 = rate_eqs(rates_old+(dt/2.)*k2, dt, q1, q2,  phimax)
            k4 = rate_eqs(rates_old+(dt)*k3, dt, q1, q2,  phimax)
            noise_vec = np.hstack((np.random.randn(N), np.zeros(N)))
            rates_old=rates_old+(dt/6.)*(k1+2*k2+2*k3+k4)+np.sqrt(dt)*sigma*noise_vec
            rates[i,0:int(num/2)]=rates_old[0:int(num/2)]
            rates[i,int(num/2):]=rates_old[N:N+int(num/2)]
        if i>49*len(time)/50:
            if i == 1 + int(49*len(time)/50):
                I = i
            rates_last[i-I,:]=rates_old
        meanr[i]=0#np.mean(f_phi(rates_old[0:N],phi_max=phimax))
        meana[i]=0#np.mean(rates_old[N:])
    return rates, rates_last, meanr, meana
    
def run_spikenetwork(T, Tprerun, init_cond, N, C, J, Jmat, t_dyn, g = 4.5, f=0.8, dt=0.01, gamma=0.5, phi_max=1e7):
    time = np.arange(0, T,dt)    
    tint = len(time)
    Nsave = 100
    spk_train_save = np.zeros((tint,Nsave))
    phi_rat_save = np.zeros((tint,Nsave))
    inp_act_save = np.zeros((tint,Nsave))
    
    time_prerun = np.arange(0, Tprerun,dt)
    for i_t, t in enumerate(time_prerun):
        if i_t==0:
            inp_act_new=np.zeros(N)
            phi_rat_new= init_cond
            spk_train_new= (1.0/dt)*(np.random.rand(N)<phi_rat_new*dt)
        else:
            inp_act_old = inp_act_new
            phi_rat_old = phi_rat_new
            spk_train_old = spk_train_new
            inp_act_new = inp_act_old-(dt/t_dyn)*inp_act_old+\
                             (dt/t_dyn)*np.dot(Jmat, spk_train_old)
            phi_rat_new = f_phi(inp_act_old, phi_max=phi_max, gamma=gamma)
            spk_train_new=(1.0/dt)*(np.random.rand(N)<phi_rat_old*dt)
    print('Prerun finished')
    for i_t, t in enumerate(time):
            inp_act_old = inp_act_new
            phi_rat_old = phi_rat_new
            spk_train_old = spk_train_new

            inp_act_save[i_t,:] = inp_act_old[0:Nsave]
            phi_rat_save[i_t,:] = phi_rat_old[0:Nsave]
            spk_train_save[i_t,:] = spk_train_old[0:Nsave]
                
            inp_act_new = inp_act_old-(dt/t_dyn)*inp_act_old+\
                             (dt/t_dyn)*np.dot(Jmat, spk_train_old)
            phi_rat_new = f_phi(inp_act_old, phi_max=phi_max, gamma=gamma)
            spk_train_new=(1.0/dt)*(np.random.rand(N)<phi_rat_old*dt)
            
    return(inp_act_save, phi_rat_save, spk_train_save, time)
    
def run_gaussnoise(T, N, C, Jmat, Delta, g = 4.5, f=0.8, dt=0.01, gamma=0.5, phi_max=1e7,  init_cond=0, Tprerun=10., verbose=True):
    '''
    Run network as in Fig.4, Francesca's paper. Rate network + external gaussian noise
    
    INPUT
    -----
    T: Simulation time
    N: Number of neurons 
    C: Number of input connections per neuron
    J: NxN array representing the connectivity matrix
    Delta: Noise intensity
    
    OUTPUT
    ------
    mu: Mean of x
    Delta_tau: autocorrelation function of x
    C_tau = autocorrelation function of phi(x)
    phi_m = mean phi(x)
    
    x_traj: trajectory of 50 neurons
    '''
    
    if init_cond ==0:
        init_cond = np.random.rand(N)
    
    pretime = np.arange(0, Tprerun, dt)
    time = np.arange(0, T, dt)
    
    #prerun
    x_t_old=init_cond        
    Jsparse  = sparse.csr_matrix(Jmat) 
    for i in range(len(pretime)-1):
        x_t_new= x_t_old+\
                    dt*(-x_t_old+np.dot(Jmat, f_phi(x_t_old, phi_max=phi_max, gamma=gamma)))\
                    +np.sqrt(2*Delta*dt)*np.random.randn(N)
        x_t_old = x_t_new
    if verbose:
        print('Prerun finished')
    #run
    xt_old=x_t_old    

    mus = np.zeros(len(time))
    phi_m = np.zeros(len(time))
    
    mus[0] = np.mean(xt_old)    
    phi_m[0] = np.mean(f_phi(xt_old, phi_max=phi_max, gamma=gamma))
    
        
    Ntr = 100
    x_traj = np.zeros((len(time),Ntr))
    x_traj[0,:]=xt_old[0:Ntr]
    for i in range(len(time)-1):
        dot_result = Jsparse.dot(f_phi(xt_old, phi_max=phi_max, gamma=gamma))
        xt_new= xt_old+\
                    dt*(-xt_old+dot_result+np.sqrt(2*Delta*dt)*np.random.randn(N))
        xt_old = xt_new
        
        x_traj[i+1,:]=xt_new[0:Ntr]
        mus[i+1] = np.mean(xt_old)
        phi_m[i+1] = np.mean(f_phi(xt_old, phi_max=phi_max, gamma=gamma))
    

    
    for n in range(Ntr):
        if n==0:
            C_tau = (1.0/Ntr)*np.correlate(f_phi(x_traj[n,:], phi_max=phi_max, gamma=gamma), f_phi(x_traj[n,:], phi_max=phi_max, gamma=gamma), mode='full')
            Delta_tau = (1.0/Ntr)*np.correlate(x_traj[n,:], x_traj[n,:], mode='full')
        else:
            C_tau += (1.0/Ntr)*np.correlate(f_phi(x_traj[n,:], phi_max=phi_max, gamma=gamma), f_phi(x_traj[n,:], phi_max=phi_max, gamma=gamma), mode='full')
            Delta_tau += (1.0/Ntr)*np.correlate(x_traj[n,:], x_traj[n,:], mode='full')
    
    return(mus, phi_m, C_tau, Delta_tau, x_traj, time)
    
def get_rightframe(ax, majory=1.0, minory=0.5, majorx=1.0, minorx=0.5, fontsize = 40, \
xlabel='', ylabel='', labelsize = 35, foursides = False, lenm=20., lenmi=10., widthm=2., widthmi=1.):
    
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)

    minorLocator = MultipleLocator(minory)
    majorLocator = MultipleLocator(majory)
    ax.yaxis.set_major_locator(majorLocator)
    ax.yaxis.set_minor_locator(minorLocator)
    majorLocator = MultipleLocator(majorx)
    minorLocator = MultipleLocator(minorx)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.xaxis.set_major_locator(majorLocator)

    if not foursides:    
        ax.spines["top"].set_visible(False)  
        ax.spines["right"].set_visible(False) 
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
    else:
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom') 

    ax.tick_params(axis='both', which='major', labelsize=labelsize,length=lenm, width=widthm)
    ax.tick_params(axis='both', which='minor', labelsize=labelsize,length=lenmi, width=widthmi)    
    
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)    
    
    return()


def Vpot(delta, delta0, J, CE, CI, g, mu, phi_max):
    from scipy.integrate import quad

    #X1 = np.linspace(-20, 20, 2000)
    #dX1 = X1[1]-X1[0]
    int0 = lambda x,z, delta, delta0, mu, phi_max: (1./np.sqrt(2*np.pi))*np.exp(-(x*x/2.0))\
                                                    *F_phi(mu+np.sqrt(delta0-delta)*x+np.sqrt(delta)*z, phi_max=phi_max)
    int1 = lambda z, delta, delta0, mu, phi_max: (1./np.sqrt(2*np.pi))*np.exp(-(z*z/2.0))\
                                                    *(quad(int0, -np.inf, np.inf, args=(z, delta, delta0, mu, phi_max))[0])**2
    integ, err = quad(int1, -np.inf,np.inf, args=(delta, delta0, mu, phi_max))
    
    int2 = lambda z,  delta, delta0, mu, phi_max: (1./np.sqrt(2*np.pi))*np.exp(-(z*z/2.0))*f_phi(mu+np.sqrt(delta0)*z, phi_max=phi_max)
    meanphi, err2 = quad(int2, -np.inf, np.inf,  args=(delta, delta0, mu, phi_max))
    Vpot = -delta**2/2.0 + J**2*(CE+g**2*CI)*(integ-delta*meanphi**2)
    return(Vpot, err, err2, meanphi, integ)
    
def bound_chaos(g_w, frac):
    mask1 = g_w<frac
    res = np.zeros(len(g_w))
    res[mask1]=1.0+g_w[mask1]
    res[~mask1] = 1-frac+2*np.sqrt(frac*g_w[~mask1])
    return(res)
    
def bound_osc(g_w, frac):
    omC = np.zeros(len(g_w))
    trans1 = np.zeros(len(g_w))
    oms = np.linspace(0,10.0, 10000)
    for i, G in enumerate(g_w):    
        trans1[i] = np.min(np.abs((-oms**2+1j*(1+frac)*oms+frac*(1+G))/(frac+1j*oms)))
        omC[i] = oms[np.argmin(np.abs((-oms**2+1j*(1+frac)*oms+frac*(1+G))/(frac+1j*oms)))]
    return(trans1, omC)
    
def bound_oscfrac(g_w, frac):
    omC = np.zeros(len(frac))
    trans1 = np.zeros(len(frac))
    oms = np.linspace(0,10.0, 10000)
    for i, F in enumerate(frac):    
        trans1[i] = np.min(np.abs((-oms**2+1j*(1+F)*oms+F*(1+g_w))/(F+1j*oms)))
        omC[i] = oms[np.argmin(np.abs((-oms**2+1j*(1+F)*oms+F*(1+g_w))/(F+1j*oms)))]
    return(trans1, omC)  
        
def my_correlate(x,y, t):
    result = np.correlate(x, y, mode='same')
    n=len(x)
    dt = t[1]-t[0]
    t = t-t[-1]
    if n/2==n/2.:
        t = t-t[(n/2)+1]
        correc = np.hstack((np.arange(np.ceil(n/2.),n+1),np.arange(np.ceil(n/2.),n-1)[::-1]))
    else:
        correc = np.hstack((np.arange(np.ceil(n/2.),n+1),np.arange(np.ceil(n/2.),n)[::-1]))
        t = t-t[n/2]
    result = result/(correc*dt)
    return result,t

def Delta(t, Dext, g_w, tau_w, J, C_E, g, C_I):
    Delta = np.zeros(len(t))
    for i, tt in enumerate(t):
        Delta[i] = Dext*((g_w**2/(2*tau_w))*np.exp(-np.abs(tt))+\
                    ((J**2*(C_E+g**2*C_I)*tau_w)/(1-tau_w**2))*(
                    np.exp(-np.abs(tt)/tau_w)-tau_w*np.exp(-np.abs(tt))))
    return Delta
def Delta0(Dext, g_w, tau_w, J, C_E, g, C_I):
    tt=0
    Delta = Dext*((g_w**2/(2*tau_w))*np.exp(-np.abs(tt))+\
                    ((J**2*(C_E+g**2*C_I)*tau_w)/(1-tau_w**2))*(
                    np.exp(-np.abs(tt)/tau_w)-tau_w*np.exp(-np.abs(tt))))
    return Delta    

def C_tau2(delta, delta0, J, CE, CI, g, mu, phi_max):
    points, weights = np.polynomial.hermite.hermgauss(100)
    if delta<0:    
        def int01(z, delta, delta0, mu, phi_max):
            F = (1/np.sqrt(np.pi))*np.sum(weights*f_phi(mu+np.sqrt(delta0-np.abs(delta))*np.sqrt(2)*points+np.sqrt(np.abs(delta))*z, phi_max=phi_max))
            return(F)
        def int02(z, delta, delta0, mu, phi_max):
            F = (1/np.sqrt(np.pi))*np.sum(weights*f_phi(mu+np.sqrt(delta0-np.abs(delta))*np.sqrt(2)*points+np.sign(delta)*np.sqrt(np.abs(delta))*z, phi_max=phi_max))
            return(F)
            
        def int1(delta, delta0, mu, phi_max):
            A = np.zeros(100)            
            F=0
            for i, a in enumerate(A):
                F+= (1/np.sqrt(np.pi))*weights[i]*(int01(np.sqrt(2)*points[i], delta, delta0, mu, phi_max))\
                   *(int02(np.sqrt(2)*points[i], delta, delta0, mu, phi_max))
            return(F)
        res = int1(delta, delta0, mu, phi_max)
    else:
        def int01(z, delta, delta0, mu, phi_max):
            F = (1/np.sqrt(np.pi))*np.sum(weights*f_phi(mu+np.sqrt(delta0-np.abs(delta))*np.sqrt(2)*points+np.sqrt(np.abs(delta))*z, phi_max=phi_max))
            return(F)
            
        def int1(delta, delta0, mu, phi_max):
            A = np.zeros(100)            
            F=0
            for i, a in enumerate(A):
               F+= (1/np.sqrt(np.pi))*weights[i]*(int01(np.sqrt(2)*points[i], delta, delta0, mu, phi_max))**2
            return(F)
        res = int1(delta, delta0, mu, phi_max)
    return(res)
    
def phi_mean(mu, delta0, J, CE, CI, g, g_w, phi_max):
    factor = (J*(CE-g*CI)-g_w)
    points, weights = np.polynomial.hermite.hermgauss(100)    
    phi_ = (1/np.sqrt(np.pi))*np.sum(weights*f_phi(mu+np.sqrt(delta0)*np.sqrt(2)*points, phi_max=phi_max))
    return(factor*phi_)
def selfconsist(mu0, dext0, J, g, CE, CI, g_w, tau_w, phi_max=3000., eps = 0.3, verbose=False):
    it = 0  
    err_ = 10.
    while err_>0.01 and it<20:
        t = np.linspace(0,50,250)
        ERR = []
        delta0 = Delta0(dext0, g_w, tau_w, J, CE, g, CI)
        mu1 = phi_mean(mu0, delta0, J, CE, CI, g, g_w, phi_max)
        delta= Delta(t, dext0, g_w, tau_w, J, CE, g, CI)
        C_t = np.zeros(len(t))
        for i,delt in enumerate(delta):
            C_t[i] = C_tau2(delt, delta0, J, CE, CI, g, mu0, phi_max)
        dext1 = np.sum(C_t*(t[1]-t[0]))
        err = [mu0-mu1, dext1 - dext0]
        if verbose:
            print(it)
            ERR.append(err)
        it+=1
        err_ = np.sqrt(err[1])
        mu0 = eps*mu1+(1-eps)*mu0
        dext0 = eps*dext1+(1-eps)*dext0
   
    return(mu0, dext0, err)
    
def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
  ''' returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF") '''
  # Starting and ending colors in RGB form
  s = hex_to_RGB(start_hex)
  f = hex_to_RGB(finish_hex)
  # Initilize a list of the output colors with the starting color
  RGB_list = [s]
  # Calcuate a color at each evenly spaced value of t from 1 to n
  for t in range(1, n):
    # Interpolate RGB vector for color at the current value of t
    curr_vector = [
      int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
      for j in range(3)
    ]
    # Add it to our list of output colors
    RGB_list.append(curr_vector)

  cc = []
  cl = color_dict(RGB_list)

  for i in range(n):
      cc.append([cl['r'][i]/255., cl['g'][i]/255., cl['b'][i]/255.])
  return cc
  
def color_dict(gradient):
  ''' Takes in a list of RGB sub-lists and returns dictionary of
    colors in RGB and hex form for use in a graphing function
    defined later on '''
  return {"hex":[RGB_to_hex(RGB) for RGB in gradient],
      "r":[RGB[0] for RGB in gradient],
      "g":[RGB[1] for RGB in gradient],
      "b":[RGB[2] for RGB in gradient]}
def hex_to_RGB(hex):
  ''' "#FFFFFF" -> [255,255,255] '''
  # Pass 16 to the integer function for change of base
  return [int(hex[i:i+2], 16) for i in range(1,6,2)]


def RGB_to_hex(RGB):
  ''' [255,255,255] -> "#FFFFFF" '''
  # Components need to be integers for hex to make sense
  RGB = [int(x) for x in RGB]
  return "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in RGB])
            
#def LIF_fn( tref, vr, vthres, mu, sigma, tm):
#
#    a = (vthres-mu)/sigma
#    b = (vr-mu)/sigma
#    print(b,a)
#    func = lambda x: np.exp(x*x)*(1.+np.sqrt(np.pi)*erf(x))    
#    partB, err = quad(func, b, a)
#    result = 1.0/((tm*partB+tref))
#    return(result, err)
#    
            
def LIF_fn( tref, vr, vthres, mu, sigma, tm, verbose=False):
    tref = tref/tm
    D= sigma**2/2.
    a = (mu-vr)/np.sqrt(2*D)
    b = (mu-vthres)/np.sqrt(2*D)
    func = lambda x: mp.exp(x*x)*mp.erfc(x)    
    partB, err = quad(func, b, a)
    result = 1.0/((np.sqrt(np.pi)*partB+tref))
    if err>100 and mu>1.0 and D<0.001:
        result = 1.0/(tref + np.log(mu/(mu-1)))
    if verbose:
        print(err)
    return(result/tm)

def spec_cross(g_w, tau_w, tau, J):
    t = np.linspace(-400,400,10000)
    dt = t[1]-t[0]
    w = np.fft.fftfreq(len(t), d = dt)
    Sx = (1+tau_w**2*(2*np.pi*w)**2)/(((1./tau)*(1-J+g_w)-tau_w*(2*np.pi*w)**2)**2
          +((tau_w/tau)*(J-1) -1)**2*(2*np.pi*w)**2)    
    
    a = np.fft.fftshift(np.fft.ifft((Sx)))
    Cx = np.real(a)
    Cx = Cx
    return(Cx,  t, Sx,  w)
    
    
    

def spec_cross2(g_w, tau_w, tau, J, Jvar, sigma, param):
    t = np.linspace(-400,400,10000)
    dt = t[1]-t[0]
    w = np.fft.fftfreq(len(t), d = dt)
    Sx1 = (1+tau_w**2*(2*np.pi*w)**2)/(((1./tau)*(1-J+g_w)-tau_w*(2*np.pi*w)**2)**2
          +((tau_w/tau)*(J-1) -1)**2*(2*np.pi*w)**2)    
    if np.max(Sx1)>1./Jvar**2:
        print('Problem')
    Sx1 = Sx1*param
    Sx = sigma**2 * Sx1 / (1- Jvar**2*Sx1)
    
    
    a = np.fft.fftshift(np.fft.ifft((Sx)))
    Cx = np.real(a)/dt
    Cx = Cx
    
    a1 = np.fft.fftshift(np.fft.ifft((sigma**2*Sx1)))
    Cx1 = np.real(a1)/dt
    Cx1 = Cx1
    return(Cx, Cx1,  t, Sx, w)

def spec_cross2_time(g_w, tau_w, tau, J, Jvar, sigma, param, t):
    dt = t[1]-t[0]
    w = np.fft.fftfreq(len(t), d = dt)
    Sx1 = (1+tau_w**2*(2*np.pi*w)**2)/(((1./tau)*(1-J+g_w)-tau_w*(2*np.pi*w)**2)**2
          +((tau_w/tau)*(J-1) -1)**2*(2*np.pi*w)**2)    
    error = 0
    if np.max(Sx1)>1./Jvar**2:
        print('Problem')
        error = 1
    Sx1 = Sx1*param
    Sx = sigma**2 * Sx1 / (1- Jvar**2*Sx1)
    
    
    a = np.fft.fftshift(np.fft.ifft((Sx)))
    Cx = np.real(a)/dt
    Cx = Cx
    
    a1 = np.fft.fftshift(np.fft.ifft((sigma**2*Sx1)))
    Cx1 = np.real(a1)
    Cx1 = Cx1/dt
    
    
    return(Cx, Cx1,  t, Sx, w, error)    
    

def corr_func(time, g_w, tau_w, tau, J):
    T = tau/tau_w;
    M = (1./tau)*np.array(((J-1, -g_w),(T, -T)))
    v, R = np.linalg.eig(M)
    dR = np.linalg.det(R)
    A1 = (1./dR)*R[1,1]*R[0,0]
    h1 = np.zeros(len(time))
    h2 = np.zeros(len(time))
    h1[time>=0] = A1*np.exp(v[0]*time[time>=0])
    A2  = -(1./dR)*R[0,1]*R[1,0]
    h2[time>=0] = A2*np.exp(v[1]*time[time>=0])
    h = h1+h2
    corr = -(np.abs(A1)**2/(v[0]+np.conj(v[0]))- A2*np.conj(A1)/(v[1]+np.conj(v[0]))) *np.exp(v[0]*np.abs(time)
            )-(np.abs(A2)**2/(v[1]+np.conj(v[1]))-A1*np.conj(A2)/(v[0]+np.conj(v[1]))) *np.exp(v[1]*np.abs(time))  
    hh_2 = np.zeros(len(corr))
    
    for i, t in enumerate(time):
        if t>=0:
            hh_2[i] = np.dot(expm(M*(time[i]+0j)), np.array((1,0)))[0]
    corr_2 = (time[1]-time[0])*np.correlate(hh_2, hh_2, mode='same')
    #Ar1 = np.sum(h1)*(time[1]-time[0])
    #Ar2 = np.sum(h2)*(time[1]-time[0])
    
    return(corr, corr_2, h, v, R, A1, A2)

def fPSD(signal, t):
    dt = (t[1]-t[0])
    T = t[-1]
    FTsignal = np.fft.fft(signal,axis = 0)
    FTfreq = np.fft.fftfreq(len(signal),d=dt)
    PSD = np.real((FTsignal*np.conj(FTsignal)))*(dt*dt/T)
    if np.shape(signal)[0]<np.size(signal):
        PSD = np.mean(PSD,1)
    mask = FTfreq>0
    return FTfreq[mask], PSD[mask]