__doc__ = '''
MixtureFit
This is an improved BEC fitting routine which fits mixtures to a bimodal model.
It can do sequential and non-sequential fitting
It requires images with corresponding parameter and independant variable file as
outputed by BECSaveWave igor routine.

Dependencies: numpy,matplotlib,lmfit,time,os,sys,argparse,pandas,numba

@author: Zachary Glassman
'''
#load necessary packages
import numpy as np
import matplotlib.pyplot as plt
from lmfit import  Model
import copy
import os
import sys
import argparse
import pandas as pd
from numba import autojit
import time
###########################
#Functions for fitting
##########################

def TF_2D(x,y,peak, Rx,Ry, centerx, centery, off, theta):
    """ 2 Dimensional Thomas-Fermi profile
    
    .. math::
            TF = A \\max\\left\\{\\left[1-\\left(\\frac{x_c}{dx}\\right)^2-\\left(\\frac{y_c}{dy}\\right)^2\\right],0\\right\\}^{3/2}
            
    :param x: array of x values
    :param y: array of y values
    :param peak: Peak value of distribution
    :param Rx: X Thomas-Fermi radius in rotated frame
    :param Ry: Y Thomas-Fermi radius in rotated frame
    :param centerx: x center in unrotated frame
    :param centery: y center in unrotated frame
    :param off: offset
    :param theta: angle of rotation
    :return: Thomas-Fermi Profile in two-dimensions
    """
    angle = np.deg2rad(theta)
    
    xcenter = (x-centerx)*np.cos(angle) - (y-centery) * np.sin(angle)
    ycenter = (x-centerx)*np.sin(angle) + (y-centery) * np.cos(angle)
    a = (np.divide((xcenter),Rx))**2
    aa = (np.divide((ycenter),Ry))**2
    bb = np.subtract(np.subtract(1, a), aa)
    c = np.zeros(bb.shape)
    b = np.power(np.maximum(bb,c),3/2)
    return (off + np.multiply(peak,b)).ravel()
    
    
def gauss_2D(x,y,peak,sigx,sigy, centerx, centery, off, theta):
    """ 2 Dimensional Gaussian profile
    
    :param x: array of x values
    :param y: array of y values
    :param peak: Peak value of distribution
    :param sigx: X variance in rotated frame
    :param sigy: Y variance in rotated frame
    :param centerx: x center in unrotated frame
    :param centery: y center in unrotated frame
    :param off: offset
    :param theta: angle of rotation
    :return: Gaussian Profile in two-dimensions
    """
    angle = np.deg2rad(theta)
    
    xcenter = (x-centerx)*np.cos(angle) - (y-centery) * np.sin(angle)
    ycenter = (x-centerx)*np.sin(angle) + (y-centery) * np.cos(angle)
    a = np.divide(np.power(xcenter,2),(2 * sigx**2))
    b = np.divide(np.power(ycenter,2),(2 * sigy**2))
    return (off + peak * np.exp(-a-b)).ravel()
    

def g2(x):
    """
    g2- summation function to enhance gaussian
    .. math::
           g_2(x) = x + \frac{x^2}{4} + \frac{x^3}{9}
    
    :param x: value
    :returns: g2(x)
    """
    return x + (x**2)/4 + (x**3)/9
    

def enhanced_gauss_2D(x,y,peak,sigx,sigy, centerx, centery, off, theta):
    """ 2 Dimensional Enhanced Gaussian profile
    
    :param x: array of x values
    :param y: array of y values
    :param peak: Peak value of distribution
    :param sigx: X variance in rotated frame
    :param sigy: Y variance in rotated frame
    :param centerx: x center in unrotated frame
    :param centery: y center in unrotated frame
    :param off: offset
    :param theta: angle of rotation
    :return: Enhanced Gaussian Profile in two-dimensions
    """
    angle = np.deg2rad(theta)
    xcenter = (x-centerx)*np.cos(angle) - (y-centery) * np.sin(angle)
    ycenter = (x-centerx)*np.sin(angle) + (y-centery) * np.cos(angle)
    a = np.divide(np.power(xcenter,2),(2 * sigx**2))
    b = np.divide(np.power(ycenter,2),(2 * sigy**2))
    return (off + peak * g2(np.exp(-a-b))/g2(1)).ravel()
    
    
def bimod_2D(x,y,centerx,centery,peakg,peaktf,Rx,Ry,sigx,sigy,off,theta):
    a = gauss_2D(x,y,peakg,sigx,sigy, centerx, centery, off/2, theta)
    b = TF_2D(x,y,peaktf, Rx,Ry, centerx, centery, off/2, theta)
    return (a + b).ravel()
    
def create_vec(shape):
    x = np.arange(0,shape[1],1)
    y = np.arange(0,shape[0],1)
    return np.meshgrid(x,y)
    
        
def pos(x,y,xc,yc,angle):
    xpos = (x-xc)*np.cos(angle) - (y-yc) * np.sin(angle)
    ypos = (x-xc)*np.sin(angle) + (y-yc) * np.cos(angle)
    return abs(xpos), abs(ypos)

@autojit
def find_rotated_mask(shape,Rx,Ry,angle,xc,yc):
    arr = np.empty((shape[0],shape[1]), dtype = bool)
    #now fill the array with distance away from center
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            xpos, ypos = pos(j,i,xc,yc,angle)
            if (xpos/Rx)**2 + (ypos/Ry)**2 <1:
                arr[i,j] = True
            else:
                arr[i,j] = False
            
    return arr

def find_mask(args, params, shape):
    """ find the mask for the TF radius"""
    
    to_rotate = {'xc': params['bimod_centerx'].value,
                 'yc': params['bimod_centery'].value,
                 'Rx': args.s*params['bimod_Rx'].value,
                 'Ry': args.s*params['bimod_Ry'].value,
                 'angle' : np.deg2rad(params['bimod_theta'].value)}
    
    if args.constant_mask:
        to_rotate['Rx'] = args.constant_mask
        to_rotate['Ry'] = args.constant_mask
        
    return find_rotated_mask(shape,**to_rotate)  
    
def subtract_back(image,n):
    back = (np.average(image[:n])+np.average(image[-n:]))/2
    return np.subtract(image,back)
    
def BEC_num(A,Rx,Ry, scalex,scaley):
    """get number of BEC atoms from fit from equation
    
    .. math::
        N = \\left(\\frac{2 \\pi}{3\\lambda^2}\\right)\\frac{2\\pi A}{5}R_x R_y
           
    :param scalex: x scale of pixel
    :param scaley: y scale of pixel
    :param A: fitted Thomas-Fermi amplitude
    :param Rx: fitted Thomas-Fermi x radius
    :param Ry: fitted Thomas-Fermi y radius
    :param sigma: optical density
    :return: atom number
    """
     
    Rx = Rx * scalex
    Ry = Ry * scaley
    sigma =  3 * (0.5891583264**2)/(2 * np.pi)
    V = 2*np.pi/5 * A* Rx * Ry
    return V/sigma 
        
def Therm_num(A,sigx, sigy, scalex,scaley):
    """get number of Therm atoms from fit from equation
        
    .. math::
        N = \\left(\\frac{2 \\pi}{3\\lambda^2}\\right)\\frac{2\\pi A}{5}R_x R_y
           
    :param scalex: x scale of pixel
    :param scaley: y scale of pixel
    :param A: fitted Gaussian amplitude
    :param Rx: fitted Gaussian x standard deviation
    :param Ry: fitted Gaussian y standard deviation
    :param sigma: optical density
    :return: atom number
    """
    Rx = sigx * scalex
    Ry = sigy* scaley
    sigma =  3 * (0.5891583264**2)/(2 * np.pi)
    V = 2*np.pi* A* Rx * Ry
    return V/sigma
        
#fancy writeout
def write_progress(step,total):
    #write out fancy
    perc_done = step/(total) * 100
    #50 character string always
    num_marks = int(.5 * perc_done)
    out = ''.join('#' for i in range(num_marks))
    out = out + ''.join(' ' for i in range(50 - num_marks))
    sys.stdout.write('\r[{0}]{1:>2.0f}%'.format(out,perc_done))
    sys.stdout.flush()
#################
#Model initialization
#################    
gauss_2d_mod = Model(gauss_2D, independent_vars = ['x','y'],prefix='gauss_')
tf_2d_mod = Model(TF_2D, independent_vars = ['x','y'],prefix='tf_')
bimod_2d_mod = Model(bimod_2D, independent_vars = ['x','y'],prefix='bimod_')

#starting parameters for bimodal fits
start_bimod_params = {'bimod_centerx': {'value':100,'min':40, 'max':200},
                      'bimod_centery':{'value':90,'min':40, 'max':200},
                      'bimod_peakg':{'value':.01,'min' : 0,'max':.5},
                      'bimod_peaktf':{'value':.2,'min' : 0,'max':.5},
                      'bimod_Rx':{'value':15 ,'min' : 0,'max':100},
                      'bimod_Ry':{'value':15,'min' : 0,'max':100},
                      'bimod_sigx':{'value':35,'min' : 0,'max':150},
                      'bimod_sigy':{'value':35,'min' : 0,'max':150},
                      'bimod_off':{'value':0 ,'min' : -1,'max': 1},
                      'bimod_theta':{'value':48, 'min' : 48, 'max': 50}
                      }


#set parameter hings in model                      
for key, value in start_bimod_params.items():
    bimod_2d_mod.set_param_hint(key, **value)       
    
##################
#Fitting
##################
def fit_image(args, data_in, filename, filepath):
    """
    function to fit image.  For a sequential fit, proceed as follows
    1. Do full bimodal fit to determine approximate TF radius
    2. Mask TF and fit to Gaussian
    3. Fix Gaussian and re-fit TF
    
    :param args: arguments passed from command line
    :param filename: filename of thing being fit
    :param filepath: path to results folder
    
    """
    data = subtract_back(data_in,20)
    x,y = create_vec(data.shape)       
    pars = bimod_2d_mod.make_params()
    
    pars['bimod_centerx'].value = args.center_x_in
    pars['bimod_centery'].value = args.center_y_in
   
    
    if args.single:
        out = bimod_2d_mod.fit(data.ravel(),pars,x=x.ravel(),y=y.ravel())
    
        report = out.fit_report()
        results =  {key:out.params[key].value for key in out.params.keys()}
        
    else:
        first_fit = bimod_2d_mod.fit(data.ravel(),pars,x=x.ravel(),y=y.ravel())
        pars = copy.deepcopy(first_fit.params)
        #now figure out mask by finding square region of larger TF radius after rotation
        mask = find_mask(args,pars,data.shape) #array for mask
        # now make maskd array
        ma = np.ma.array(data, mask = mask)
        #now we apply the same mask to the vectors
        xm = np.ma.array(x, mask = mask) 
        ym = np.ma.array(y, mask = mask)
        #now fix TFpeak to 0 and fix center
        TF_val = pars['bimod_peaktf'].value
        pars['bimod_peaktf'].value = 0
        pars['bimod_peaktf'].vary = False
        pars['bimod_Rx'].vary = False
        pars['bimod_Ry'].vary = False
        #set gaussian wings to good guesses
        pars['bimod_sigx'].value = 10
        pars['bimod_sigy'].value = 10
        pars['bimod_peakg'].value = 0.1
        #fit
        second_fit = bimod_2d_mod.fit(ma.compressed(),
                                      pars,
                                      x=xm.compressed(),
                                      y=ym.compressed())
        
        pars = copy.deepcopy(second_fit.params)
        #now free the TF parameters
        pars['bimod_peaktf'].value = TF_val
        pars['bimod_peaktf'].vary = True
        pars['bimod_Rx'].vary = True
        pars['bimod_Ry'].vary = True
        #fix gaussian parameters
        pars['bimod_sigx'].vary = False
        pars['bimod_sigy'].vary = False
        pars['bimod_peakg'].vary = False
        
        #do third fit
        out = bimod_2d_mod.fit(data.ravel(),pars,x=x.ravel(),y=y.ravel())
        #results
        report = out.fit_report()
        results =  {key:out.params[key].value for key in out.params.keys()}
        
    if args.pretty_print:  
        try:
            data_out = bimod_2d_mod.eval(params = pars,
                                         x = x,
                                         y = y).reshape(data.shape[0],
                                                        data.shape[1])
            xs = np.arange(0,data.shape[1],1)
            ys = np.arange(0,data.shape[0],1)
       
            plt.clf()
            plt.title(filename)
            ax1 = plt.subplot2grid((3,3), (0,0))
            ax2 = plt.subplot2grid((3,3), (0,1))
            axm = plt.subplot2grid((3,3),(0,2))
            ax3 = plt.subplot2grid((3,3), (1, 0), colspan=3)
            ax4 = plt.subplot2grid((3,3), (2, 0), colspan=3)
        
            ax1.imshow(data)
            ax2.imshow(data_out)
            if args.single:
                pass
            else:
                axm.imshow(ma)
            ax3.scatter(xs,np.sum(data, axis = 0),s=5,c='orange')
            ax3.plot(np.sum(data_out, axis = 0))
            ax4.plot(np.sum(data_out, axis = 1))
            ax4.scatter(ys,np.sum(data, axis = 1),s=5, c='orange')
            if args.single:
                pass
            else:
               xsm = np.arange(0,ma.shape[1],1)
               ysm = np.arange(0,ma.shape[0],1)
               data_outm = bimod_2d_mod.eval(params = second_fit.params,
                                             x = x,
                                             y = y).reshape(data.shape[0],
                                                            data.shape[1])
               ax3.plot(np.sum(data_outm, axis = 0))
               ax4.plot(np.sum(data_outm, axis = 1))
               ax3.scatter(xsm,np.sum(ma, axis = 0),s=5,c='red')
               ax4.scatter(ysm,np.sum(ma, axis = 1),s=5,c='red') 
               
            ax1.set_title('Data')
            ax2.set_title('Fitted Data')
            ax3.set_title('X Sum')
            ax4.set_title('Y Sum')
            plt.tight_layout()
            plt.savefig(os.path.join(filepath,filename + '.png'),dpi = 200)
        except:
            report += 'Had trouble making plots'
       
    return report, results

def main(args):
    """main routine"""
    start = time.time()
    filepath = os.path.join(os.getcwd(),args.path) 
    files = [f for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath,f))]
    #filter for .pxp files (probably there)
    files = [f for f in files if f[-3:] != "pxp"]
    if args.n:
        files = files[:args.n]
    results_path = os.path.join(filepath,'Results')
    if not os.path.exists(results_path): 
        os.makedirs(results_path)
    #create dataframe
    pd_index = [f.rstrip('.txt') for f in files]
    df = pd.DataFrame(columns = list(start_bimod_params.keys()),
                      index = pd_index)
   
    #fit the suckers
    index = 0
    tot_index = len(files)
    err_list = []
    print('Commencing fit of {0} files'.format(tot_index))
    write_progress(index,tot_index)
    with open(os.path.join(results_path,'Fit_Results.txt'),'w') as f:
        for i in files:
            name = i.rstrip('.txt')
            try:
                data = np.loadtxt(os.path.join(filepath,i))
                report, results = fit_image(args,data,name,results_path)
                
                
            except:
                report = 'Unable to fit, possibly invalid file type'
                results = False
                err_list.append(i)
            
            #write report to file
            f.write('{0}\n{1}\n{2}\n'.format(i, report,''.join('#' for i in range(30))))
            #update pandas array
            if results:
                df.loc[name] = pd.Series(results)
            index += 1
            write_progress(index,tot_index)
            
    #now calculate NBEC and NTHERM from Results and save
    df['N_BEC'] = np.vectorize(BEC_num)(df['bimod_peaktf'],
                      df['bimod_Rx'],
                      df['bimod_Ry'], args.scalex,args.scaley)
    df['N_Therm'] = np.vectorize(Therm_num)(df['bimod_peakg'],
                      df['bimod_sigx'],
                      df['bimod_sigy'], args.scalex,args.scaley)
                      
    df.to_csv(os.path.join(results_path,'Param_Results.txt'))
    end = time.time()  
    print()
    print_str = 'Completed fitting {0} out of {1} files'
    print(print_str.format(tot_index-len(err_list),tot_index))
    print('Total time: {0}\n Time per image: {1}'.format(end-start,
                                                      (end-start)/tot_index))
    if len(err_list) > 0:
        print('Error Files are')
        for i in err_list:
            print('  ' + i)
    
        
      
    
if __name__  == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('-v', action = 'store_true',
                       dest = 'pretty_print',
                       default = False,
                       help = 'Generate visual file for each fit (default False)')
                       
   parser.add_argument('-single', action = 'store_true',
                       dest = 'single',
                       default = False,
                       help = 'Do a single bimodal fit (default False)')
                       
   parser.add_argument('-d', action = 'store_true',
                       dest = 'debug',
                       default = False,
                       help = 'Debug code (default False)')
                       
   parser.add_argument('-p', action = 'store',
                      dest = 'path',
                      default = '',
                      help = 'Relative path for folder containing files')
   
   parser.add_argument('-cm', action = 'store',
                       dest = 'constant_mask',
                       default = False,
                       help = 'Use a constant mask (entered in matrix units) (default false)')
                       
   parser.add_argument('-s', action = 'store',
                       dest = 's',
                       default = 1,
                       type = float,
                       help = 'Overestimate parameter of mask, only applies for non-constant mask (default = 1)')
        
   parser.add_argument('-n', action = 'store',
                       dest = 'n',
                       type = int,
                       default = False,
                       help = 'Fit only first n images (default False)')
                       
   parser.add_argument('-sx', action = 'store',
                       dest = 'scalex',
                       default = 7.04,
                       type = float,
                       help = 'x scale of pixel (default 7.04)') 
                       
   parser.add_argument('-sy', action = 'store',
                       dest = 'scaley',
                       default = 7.04,
                       type = float,
                       help = 'y scale of pixel (default 7.04)') 
   parser.add_argument('-cx', action = 'store',
                       dest = 'center_x_in',
                       type = float,
                       default = 100,
                       help = 'X center in matrix units (default )')
                       
   parser.add_argument('-cy', action = 'store',
                       dest = 'center_y_in',
                       type = float,
                       default = 90,
                       help = 'Y center in matrix units (default )')
                       
   results = parser.parse_args()
   
   main(results)

