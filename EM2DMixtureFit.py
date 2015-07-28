# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 13:27:04 2015
Full 2D EM fitting
@author: zag
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')



def norm_dist(x,y,mu_x,mu_y,sigsqr_x,sigsqr_y):
    a = -(x-mu_x)**2/(2*sigsqr_x)
    b = -(y-mu_y)**2/(2*sigsqr_y)
    return 1/(2*np.pi * np.sqrt(sigsqr_x*sigsqr_y))*np.exp(a+b)

def tf_dist(x,y,mu_x,mu_y,sigsqr_x,sigsqr_y):
    """compute a thomas fermi distribution normalized"""
    a = (x-mu_x)**2/sigsqr_y
    b = (y-mu_y)**2/sigsqr_y
    ans = (1-a-b)**(3/2)
    out = np.zeros(len(ans))
    return np.maximum(ans,out)
    
def format_params(params):
    ans = ''
    for key in params.keys():
        ans = ans + '{0} = {1:>.4f}\n'.format(key,params[key])
    return ans
        
def e_step(data,qc, mu,sigsqrT,sigsqrN,q):
    """in this step we have fixed parameters and we solve for the posterior
    distribution qcT and qcN for the hidden variables (class labels)"""
    norm = (1-q)*norm_dist(data,mu,sigsqrN)
    tf =   q*tf_dist(data,mu,sigsqrT)
    denom = tf+norm
    qc['T'] = tf/denom
    qc['N'] = norm/denom
    
    return qc

      
def m_step(data,qc, mu,sigsqrT,sigsqrN,q):
    """now we compute the new parameters with the given posterior distributions"""
    mu_t = (data * qc['T']).sum()/qc['T'].sum()
    sigsqrT_new = 3 *(data-mu_t)**2 * qc['T']
    sigsqrN_new = (data-mu_t)**2 * qc['N']
    #now normalize

    sigsqrT_new = sigsqrT_new.sum()/qc['T'].sum()
    sigsqrN_new = sigsqrN_new.sum()/qc['N'].sum()
    
    q = 0.99 #- qc['N'].sum()/len(qc['N'])
    improvement = max(np.abs(sigsqrT_new-sigsqrT),np.abs(sigsqrN_new-sigsqrN))
    return improvement, {'mu':mu_t,'sigsqrT':sigsqrT_new,'sigsqrN':sigsqrN_new,'q':q}

def subtract_back(image,n):
    """subtract average of n rows of top and bottom from background"""
    back = (np.average(image[:n])+np.average(image[-n:]))/2
    
    return np.subtract(image,back)
    

def get_data(filename):
    """get data and turn into probability distribution"""
    data = np.loadtxt(filename)
    data_in = np.sum(subtract_back(data,20),axis = 0)
    data = data_in[60:120]
    xk = np.arange(len(data))
    pk = data/np.sum(data)
    dist = stats.rv_discrete(name = 'bimod' , values = (xk,pk))
    
    return dist.rvs(size = 100000), pk, data_in, np.sum(data)
    
def main():
    sim_data,data,raw_data,sum_data = get_data('C:\\Users\zag\\Documents\\BECFitting\\2015-7-14\\7-14-matrix0104_0.txt')
    #need to keep track of qc for both TF and Normal components
    qc = {'T':np.zeros(len(sim_data)),'N':np.zeros(len(sim_data))}
    params = {'mu':np.mean(sim_data),'sigsqrT':9**2,'sigsqrN':25**2,'q':.5}
    dif = 1
    eps = 1e-20
    n_iter = 0
    #while dif > eps:
    for _ in range(100):
        qc = e_step(sim_data,qc,**params)
        #print((1-params['q'])*np.log(norm_dist(data,params['mu'],params['sigsqrN'])).sum()+params['q']*np.log(tf_dist(data,params['mu'],params['sigsqrT'])).sum())
        dif, params = m_step(sim_data,qc,**params)
        n_iter += 1
    
        
    print('Number of iterations: {0}'.format(n_iter))
    print(format_params(params))
    print('Thomas Fermi radius is : {0}'.format(np.sqrt(params['sigsqrT'])))
    print('Gaussian 1/e^2 radius is : {0}'.format(np.sqrt(params['sigsqrN'])))
    mu = params['mu']+60
    sigsqrN = params['sigsqrN']
    sigsqrT = params['sigsqrT']
    q = params['q']

    x = np.arange(len(raw_data))
    norm = (1-q)*norm_dist(x,mu,sigsqrN)
    tf =   q*tf_dist(x,mu,sigsqrT)
    
    fig, ax = plt.subplots(2,1)
    ax[0].hist(sim_data+60,normed = True, bins = len(data))
    #ax[0].plot(x,data,'o-',label = 'data')
    ax[0].plot(x,tf,label = 'TF')
    ax[0].plot(x,norm,label = 'Norm')
    ax[0].plot(x, norm+tf,label = 'Fitted')
    ax[1].plot(x,raw_data,'o-')
    ax[1].plot(x,(norm+tf)*sum_data)
    ax[0].legend()
    plt.show()
    
   
if __name__ == '__main__':
    main()