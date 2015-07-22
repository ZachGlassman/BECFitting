# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 21:15:12 2015
MixtureSimulation
This will create a series of images and the corresponding Pandas-dataframe
for comparison with different fitting parameters.  Will use same 
functions as MixtureFit
@author: zag
"""
import numpy as np
import pandas as pd
from MixtureFit import bimod_flat_2d_mod ,create_vec,find_rotated_mask
import os

def find_mask(params, shape):
    """ find the mask for the TF radius"""
    
    to_rotate = {'xc': params['bimod_centerx'].value,
                 'yc': params['bimod_centery'].value,
                 'Rx': params['bimod_Rx'].value,
                 'Ry': params['bimod_Ry'].value,
                 'angle' : np.deg2rad(params['bimod_theta'].value)}
    
        
    return find_rotated_mask(shape,**to_rotate)  
    
def create_image(name,
                 bimod_centerx,
                 bimod_centery,
                 bimod_peakg,
                 bimod_peaktf,
                 bimod_Rx,
                 bimod_Ry,
                 bimod_sigx,
                 bimod_sigy,
                 bimod_off,
                 bimod_theta):
    """function to create image with parameters given as before"""
    shape = (142,177)
    #make the paramters
    pars = bimod_flat_2d_mod.make_params(
                      bimod_centerx=bimod_centerx,
                      bimod_centery=bimod_centery,
                      bimod_peakg=bimod_peakg,
                      bimod_peaktf=bimod_peaktf,
                      bimod_Rx=bimod_Rx,
                      bimod_Ry=bimod_Ry,
                      bimod_sigx=bimod_sigx,
                      bimod_sigy=bimod_sigy,
                      bimod_off=bimod_off,
                      bimod_theta=bimod_theta)
    #create vectors to broadcast over
    x,y = create_vec(shape) 
    #create mask
    mask = find_mask(pars,shape)
    data = bimod_flat_2d_mod.eval(params = pars,
                                         x = x.ravel(),
                                         y = y.ravel(),
                                         mask = mask).reshape(shape[0],
                                                        shape[1])
    np.savetxt(name, data)

def main():
    #get correct directory
    path = os.path.join(os.getcwd(),'Simulation')
    os.chdir(path)
    #make dataframe
    files = ['{0}test.txt'.format(i) for i in range(100)]
  
    param_names = [   'bimod_centerx',
                      'bimod_centery',
                      'bimod_peakg',
                      'bimod_peaktf',
                      'bimod_Rx',
                      'bimod_Ry',
                      'bimod_sigx',
                      'bimod_sigy',
                      'bimod_off',
                      'bimod_theta']
                      
    df = pd.DataFrame(columns = param_names,index = files)
    #now we will generate random images with each parameter in a normal range
    ranges = {}
    ranges['bimod_centerx'] = (87,90)
    ranges['bimod_centery'] = (87,90)
    ranges['bimod_sigx'] = (15,25)
    ranges['bimod_sigy'] = (15,25)
    ranges['bimod_peakg'] = (0,0.05)
    ranges['bimod_peaktf'] = (0.02,0.5)
    ranges['bimod_Rx'] = (9,14)
    ranges['bimod_Ry'] = (9,14)
    ranges['bimod_theta'] = (48,49)
    ranges['bimod_off'] = (-0.001,0.001)
    
   
    for name in files:
        results = {key:np.random.uniform(*ranges[key]) for key in param_names}
        #now add to pandas df
        df.loc[name] = pd.Series(results)
        create_image(name,**results)
      
    df.to_csv('Sim_params.txt')
    print('Done')

if __name__ == '__main__':
    main()
    

