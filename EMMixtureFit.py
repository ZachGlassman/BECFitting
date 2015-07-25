# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 18:09:45 2015
EM Mixture Fit

This will use a variant of the EM algorithm to fit the mixture data.
We will create a probability distribution from the measured density
then iterately maximize the log-likelihood of the components.
We assume here that mu is the same for both of the bimodal distributions
@author: zag
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')


#first define normalized probability distributions
def norm_dist(x,mu,sigsqr):
    return 1/(np.sqrt(sigsqr*2*np.pi))*np.exp(-(x-mu)**2/(2*sigsqr))

def tf_dist(x,mu,sigsqr):
    ans = 3/(4*np.sqrt(sigsqr))*(1-(x-mu)**2/sigsqr)
    return np.maximum(ans,np.zeros(ans.shape[0]))
    

def e_step(data,qc, mu,sigsqrT,sigsqrN,q):
    """in this step we have fixed parameters and we solve for the posterior 
    distribution qcT and qcN for the hidden variables (class labels)"""
    norm = (1-q)*norm_dist(data,mu,sigsqrN)
    tf =   q*tf_dist(data,mu,sigsqrT)
    #tf = q * norm_dist(data,mu,sigsqrT)
    denom = tf+norm
    qc['T'] = tf/denom
    qc['N'] = norm/denom
    
    return qc
    
def m_step(data,qc, mu,sigsqrT,sigsqrN,q):
    """now we compute the new parameters with the given posterior distributions"""
  
    sigsqrT_new = 3*(mu-data)**2 * qc['T']
    #sigsqrT_new = (data-mu)**2 * qc['T']
    sigsqrN_new = (data-mu)**2 * qc['N']
    #now normalize
    sigsqrT_new = sigsqrT_new.sum()/qc['T'].sum()
    sigsqrN_new = sigsqrN_new.sum()/qc['N'].sum()
    #q = qc['T'].sum()/data.size
    improvement = max(np.abs(sigsqrT_new-sigsqrT),np.abs(sigsqrN_new-sigsqrN))
    
    return improvement, {'mu':mu,'sigsqrT':sigsqrT_new,'sigsqrN':sigsqrN_new,'q':q}

def subtract_back(image,n):
    """subtract average of n rows of top and bottom from background"""
    back = (np.average(image[:n])+np.average(image[-n:]))/2
    return np.subtract(image,back)
    
def get_data(filename):
    """get data and turn into probability distribution"""
    data = np.loadtxt(filename)
    data = np.sum(subtract_back(data,20),axis = 0)
    data = data + np.abs(np.min(data))
    xk = np.arange(len(data))
    pk = data/np.linalg.norm(data)
    pk = pk/np.sum(pk)
    dist = stats.rv_discrete(name = 'bimod' , values = (xk,pk))
    return dist.rvs(size = 10000),pk
    
def main():
    sim_data,data = get_data('C:\\Users\zag\\Documents\\BECFitting\\2015-7-14\\7-14-matrix0108_0.txt')
    #need to keep track of qc for both TF and Normal components
    qc = {'T':np.zeros(len(sim_data)),'N':np.zeros(len(sim_data))}
    params = {'mu':np.mean(sim_data),'sigsqrT':5,'sigsqrN':30,'q':.5}
    dif = 1
    eps = .001
    #while dif > eps:
    for _ in range(100):
        qc = e_step(sim_data,qc,**params)
        dif, params = m_step(sim_data,qc,**params)
    print(params)
    

    mu = params['mu']
    sigsqrN = params['sigsqrN']
    sigsqrT = params['sigsqrT']
    q = params['q']
    
    x = np.arange(len(data))
    norm = (1-q)*norm_dist(x,mu,sigsqrN)
    #tf =   q*tf_dist(x,mu,sigsqrT)
    tf = q * norm_dist(x,mu,sigsqrT)
    

    #plt.hist(sim_data,normed = True, bins = len(data))
    plt.plot(x,data,'o-',label = 'data')
    plt.plot(x, norm+tf,label = 'Fitted')
    plt.show()
    
   


if __name__ == '__main__':
    
    main()