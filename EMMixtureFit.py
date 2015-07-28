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
import sys


#first define normalized probability distributions
def norm_dist(x,mu,sigsqr):
    return 1/(np.sqrt(sigsqr*2*np.pi))*np.exp(-(x-mu)**2/(2*sigsqr))

def tf_dist(x,mu,sigsqr):
    """compute a thomas fermi distribution with a guassian of width 1/100 of
    the tf radius and return normalized result."""
    ans = 1-(x-mu)**2/sigsqr
    out = np.zeros(len(ans))
    return  3/(4*np.sqrt(sigsqr))*np.maximum(ans,out)
    
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

      
def m_step(m,data,qc, mu,sigsqrT,sigsqrN,q):
    """now we compute the new parameters with the given posterior distributions"""
    mu_t = (data * qc['T']).sum()/qc['T'].sum()
    sigsqrT_new = m *(data-mu_t)**2 * qc['T']
    sigsqrN_new = (data-mu_t)**2 * qc['N']
    #now normalize

    sigsqrT_new = sigsqrT_new.sum()/qc['T'].sum()
    sigsqrN_new = sigsqrN_new.sum()/qc['N'].sum()
    
    q = qc['T'].sum()/len(qc['T'])
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
    
def do_em(sim_data,qc,params,niter,m, eps = None):
    if niter == False:
        n_iter = 0
        dif = 1
        while dif > eps:
            qc = e_step(sim_data,qc,**params)
            dif, params = m_step(m,sim_data,qc,**params)
            n_iter += 1
    else:
        for _ in range(niter):
            qc = e_step(sim_data,qc,**params)
            dif, params = m_step(m,sim_data,qc,**params)
            
    return params
    
#fancy writeout
def write_progress(step,total,string = None):
    """write the progress out to the window"""
    perc_done = step/(total) * 100
    #50 character string always
    num_marks = int(.5 * perc_done)
    out = ''.join('#' for i in range(num_marks))
    out = out + ''.join(' ' for i in range(50 - num_marks))
    sys.stdout.write('\r[{0}]{1:>2.0f}% {2}'.format(out,perc_done,string))
    sys.stdout.flush()
    
def main():
    sim_data,data,raw_data,sum_data = get_data('C:\\Users\zag\\Documents\\BECFitting\\2015-7-14\\7-14-matrix0103_0.txt')
    #need to keep track of qc for both TF and Normal components
    qc = {'T':np.zeros(len(sim_data)),'N':np.zeros(len(sim_data))}
    params = {'mu':np.mean(sim_data),'sigsqrT':9**2,'sigsqrN':25**2,'q':.5}
    ans = {}
    m = np.linspace(2.5,9,100)
    index = 0
    tot_index = len(m)
    write_progress(index,tot_index,m[0])
    for j in m:
        ans[j] = do_em(sim_data,qc,params.copy(),50,j,1e-20)
        index += 1
        write_progress(index,tot_index,m[index-1])
        
    fig, ax = plt.subplots(4,1)
    x = np.arange(len(raw_data))
    ax[0].hist(sim_data+60,normed = True, bins = len(data))
    ax[1].plot(x,raw_data,'o-')
    minj = [0,100000]
    for j in ans.keys():
        mu = ans[j]['mu']+60
        sigsqrN = ans[j]['sigsqrN']
        sigsqrT = ans[j]['sigsqrT']
        q = ans[j]['q']
        
        norm = (1-q)*norm_dist(x,mu,sigsqrN)
        tf =   q*tf_dist(x,mu,sigsqrT)
        ans[j]['chisqr'] = (((norm+tf)*sum_data - raw_data)**2).sum()
        if ans[j]['chisqr']  < minj[1]:
            minj[0] = j
            minj[1] = ans[j]['chisqr']
     
    j = minj[0]
    print('\nOptimal scaling is : {0}'.format(j))
    print('TF Radius: {0}'.format(np.sqrt(ans[j]['sigsqrT'])))
    print('Gauss width: {0}'.format(np.sqrt(ans[j]['sigsqrN'])))
    mu = ans[j]['mu']+60
    sigsqrN = ans[j]['sigsqrN']
    sigsqrT = ans[j]['sigsqrT']
    q = ans[j]['q'] 
    norm = (1-q)*norm_dist(x,mu,sigsqrN)
    tf =   q*tf_dist(x,mu,sigsqrT)       
    ax[0].plot(x,tf,label = 'TF {0}'.format(j))
    ax[0].plot(x,norm,label = 'Norm {0}'.format(j))
    ax[0].plot(x, norm+tf,label = 'Fitted {0}'.format(j))
    ax[1].plot(x,(norm+tf)*sum_data)
        
    ax[2].plot(m,[ans[i]['chisqr'] for i in m])    
    ax[3].plot(m,[np.sqrt(ans[i]['sigsqrT']) for i in m])
    #ax[0].legend()
    plt.tight_layout()
    plt.show()

   
if __name__ == '__main__':
    main()