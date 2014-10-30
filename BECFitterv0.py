__doc__ = '''
BECFitter v0.0
This is an improved BEC fitting routine which fits mixtures to a bimodal model.
It requires images with corresponding parameter and independant variable file as
outputed by BECSAveWave igor routine.

Dependencies: numpy,matplotlib,lmfit,time,os,functools

Classes: fit_obj, fit_storage
@author: Zachary Glassman
'''
#load necessary packages
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, Parameter, report_fit
import time
import os
import functools

###########################
#Functions for fitting
##########################

#TF_2D
#returns the Thomas fermi function for 2 dimensions
#input params,x,y
#output thomas-fermi    
def TF_2D(params,x,y):   
    peak = params['TFpeak'].value 
    Rx = params['TFRx'].value 
    Ry = params['TFRy'].value 
    centerx = params['TFcenterx'].value
    centery = params['TFcentery'].value
    off = params['offset'].value 
    angle = np.deg2rad(params['theta'].value)
    
    xcenter = (x-centerx)*np.cos(angle) - (y-centery) * np.sin(angle)
    ycenter = (x-centerx)*np.sin(angle) + (y-centery) * np.cos(angle)
    a = (np.divide((xcenter),Rx))**2
    aa = (np.divide((ycenter),Ry))**2
    bb = np.subtract(np.subtract(1, a), aa)
    c = np.zeros(bb.shape)
    b = np.power(np.maximum(bb,c),3/2)
    return(off + np.multiply(peak,b))
    
    
#gaussian_2D
#Returns the gaussian function for 2 dimensinos
#inputs params, x,y
#outputs gaussian
def gauss_2D(params,x,y):
    off = params['offset'].value
    peak = params['Gpeak'].value 
    sigx = params['sigx'].value 
    sigy = params['sigy'].value
    centerx = params['Gcenterx'].value
    centery = params['Gcentery'].value
    angle = np.deg2rad(params['theta'].value)
    
    xcenter = (x-centerx)*np.cos(angle) - (y-centery) * np.sin(angle)
    ycenter = (x-centerx)*np.sin(angle) + (y-centery) * np.cos(angle)
    a = np.divide(np.power(xcenter,2),(2 * sigx**2))
    b = np.divide(np.power(ycenter,2),(2 * sigy**2))
    return(off + peak * np.exp(-a-b))
    
#g2- summation function to enhance gaussian
def g2(x):
    return(x + (x**2)/4 + (x**3)/9) 
    
#enhanced_gaussian_2D
#Returns the enhanced gaussian function for 2 dimensinos
#inputs params, x,y
#outputs gaussian
def enhanced_gauss_2D(params,x,y):
    off = params['offset'].value
    peak = params['Gpeak'].value 
    sigx = params['sigx'].value 
    sigy = params['sigy'].value
    centerx = params['Gcenterx'].value
    centery = params['Gcentery'].value
    angle = np.deg2rad(params['theta'].value)
    xcenter = (x-centerx)*np.cos(angle) - (y-centery) * np.sin(angle)
    ycenter = (x-centerx)*np.sin(angle) + (y-centery) * np.cos(angle)
    a = np.divide(np.power(xcenter,2),(2 * sigx**2))
    b = np.divide(np.power(ycenter,2),(2 * sigy**2))
    return(off + peak * g2(np.exp(-a-b))/g2(1))
    
    
def enhanced_bimod_2D(params,x,y):
    return(enhanced_gauss_2D(params,x,y) + TF_2D(params,x,y)) 
 
 

#enhanced_bimod2min
#fit the bimodial distribution to some data 
#A, B inital guesses for gaussian and TF respectively
def enhanced_bimod2min(params,x,y,data):
    resid = enhanced_bimod_2D(params,x,y).ravel() - data
    return(resid)
    
 

############################
#object to store igor output
###########################

class fit_obj(object):
    def __init__(self,name):
        self.images = False
        self.name = name
        self.results = None
        
    
    def pretty_print(self):
        print("Name: ",self.name)
        for i in self.params:
            print(i.value)
            #print(i.name + ":",i.value)
            
    def process_image(self):
        self.images = True
        self.get_image()
        self.subtract_background(20)
      
    def get_image(self):
        self.init_image = np.transpose(np.loadtxt(self.name))
      
    def show_image(self):
        plt.figure()
        ax0 = plt.subplot2grid((3,2), (0,0), colspan=1)
        ax1 = plt.subplot2grid((3,2),(0,1), colspan=1)
        ax2 = plt.subplot2grid((3,2), (1, 0), colspan=2)
        ax3 = plt.subplot2grid((3,2),(2,0), colspan = 2)
        ax0.imshow(self.init_image)
        ax0.set_title('Initial Image')
        ax1.imshow(self.image)
        ax1.set_title('Background Subtracted')
        x = np.linspace(0,len(self.image[0])/7.04,len(self.image[0]))
        ax2.plot(x,np.sum(self.image, axis = 0))
        ax3.plot(np.sum(self.image, axis = 1))
        ax2.set_title('X Sum')
        ax3.set_title('Y Sum')
        
    def show_fit_results(self):
        plt.figure()
        ax0 = plt.subplot2grid((3,2), (0,0), colspan=1)
        ax1 = plt.subplot2grid((3,2),(0,1), colspan=1)
        ax2 = plt.subplot2grid((3,2), (1, 0), colspan=2)
        ax3 = plt.subplot2grid((3,2),(2,0), colspan = 2)
        fitim = enhanced_bimod_2D(self.params,self.x,self.y)
        ax0.imshow(self.image)
        ax0.set_title('Initial Image')
        ax1.imshow(fitim)
        ax1.set_title('Fitted Image')
        ax2.plot(np.sum(self.image, axis = 0))
        ax2.plot(np.sum(fitim,axis = 0))
        ax3.plot(np.sum(self.image, axis = 1))
        ax3.plot(np.sum(fitim,axis = 1))
        ax2.set_title('X Sum')
        ax3.set_title('Y Sum')
        
    def show_temp_results(self):
        plt.figure()
        ax0 = plt.subplot2grid((3,2), (0,0), colspan=1)
        ax1 = plt.subplot2grid((3,2),(0,1), colspan=1)
        ax2 = plt.subplot2grid((3,2), (1, 0), colspan=2)
        ax3 = plt.subplot2grid((3,2),(2,0), colspan = 2)
        fitim = enhanced_bimod_2D(self.params,self.x,self.y)
        ax0.imshow(self.temp_image)
        ax0.set_title('Initial Image')
        ax1.imshow(fitim)
        ax1.set_title('Fitted Image')
        ax2.plot(np.sum(self.temp_image, axis = 0))
        ax2.plot(np.sum(fitim,axis = 0))
        ax3.plot(np.sum(self.temp_image, axis = 1))
        ax3.plot(np.sum(fitim,axis = 1))
        ax2.set_title('X Sum')
        ax3.set_title('Y Sum')
    
    def subtract_background(self,n):
        if self.images == False:
            return()
        back = (np.average(self.init_image[:n])+np.average(self.init_image[-n:]))/2
        self.image = np.subtract(self.init_image,back)
    
    def create_axis(self):
        dim = self.image.shape
        X = np.linspace(0,dim[1],dim[1])
        Y = np.linspace(0,dim[0],dim[0])
        x1,y1 = np.meshgrid(X,Y)
        self.x = np.multiply(x1,7.04)
        self.y = np.multiply(y1,7.04)
        
    def initialize_params(self):
        ###find center
        idx = np.argmax(self.image,axis = None)
        multi_idx = np.unravel_index(idx,self.image.shape)
        #initialize parameter object
        self.params = Parameters()
        self.params.add('TFpeak', value = .2, min = 0.0)
        self.params.add('Gpeak', value = 0.01, min = 0.0)
        self.params.add('TFRx', value = 127,min = 122,max = 132)
        self.params.add('TFRy', value = 127,min = 122, max = 132)
        self.params.add('TFcenterx', value = multi_idx[0] * 7.04)
        self.params.add('TFcentery', value = multi_idx[1] * 7.04)
        self.params.add('sigx', value = 20 * 7,min = 0)
        self.params.add('sigy', expr = 'sigx')
        self.params.add('Gcenterx',expr = 'TFcenterx')
        self.params.add('Gcentery', expr = 'TFcentery')
        self.params.add('offset',value = 0)
        self.params.add('theta',value = 48, vary = False)
        
    #default fit type is levenberg marquardt, if argument is passed in,
    # it can perform any fitting methods offerred in scipy.optimize
    def fit_enhanced_bimod(self,data,fittype = 'leastsq' ):
        x = self.x
        y = self.y
        data1 = data.ravel()
        self.results = minimize(enhanced_bimod2min, self.params, args = (x,y,data1),method = fittype)
    
    def report_last_results(self): 
        if self.results == None:
            print('Fit not yet performed')
        else:
            report_fit(self.results)
            self.num_atoms()
            print('BECNum:',self.bec_num)
            print('THERMNum:',self.therm_num)
            
    def num_atoms(self):
        sig = 3 * (0.5891583264**2)/(2 * np.pi)
        Rx = self.params['TFRx'].value 
        Ry = self.params['TFRy'].value
        Sx = self.params['sigx'].value
        Sy = self.params['sigy'].value
        TP = self.params['TFpeak'].value
        GP = self.params['Gpeak'].value
        bec = TP*(2.0/5.0)*Rx*Ry*1j*((-1j*np.pi/2+np.log(2*Rx*Ry*Ry))-(1j*np.pi/2+np.log(2*Rx*Ry*Ry)))
        therm = GP*(251.0/147.0) *np.pi*Sx*Sy
        self.bec_num = np.real(bec)/sig
        self.therm_num = therm/sig
        
    #Function which takes string of inputs, one for each variable and each is 
    #either 1 or 0 for binary free variable. 0 is fix variable
    #string must be ordered in same way as parameters declared
    def params_vary(self,string_in):  
        if len(self.params) != len(string_in):
            print('Invalid String argument')
            pass
        else:
            k = 0
            for i in self.params:
                if string_in[k] == '1':
                    self.params[i].vary = True
                else:
                    self.params[i].vary = False
                k = k + 1
            
    #function to help conserve memory
    def clear_images(self):
        self.images = False
        self.image = None
        self.init_image = None
        
    def prepare_fit(self):
        self.process_image()
        self.initialize_params()
        self.create_axis()
        self.params_vary('101111000010')
        
    def readout(self):
        out = []
        for i in self.params:
            out.append([self.params[i].name,self.params[i].value])
        out.append(['BECNUM',int(self.bec_num)])
        out.append(['THERMNUM',int(self.therm_num)])
        out.append(['TOTNUM',int(self.bec_num+self.therm_num)])
        return(dict(out))
        
    def subtract_bec(self,S):
        self.params['Gpeak'].value = 0
        Rx = self.params['TFRx'].value 
        Ry = self.params['TFRy'].value
        self.params['TFRx'].value = Rx * S
        self.params['TFRy'].value = Ry * S
        self.temp_image = self.image - TF_2D(self.params,self.x,self.y)
        self.params['TFRx'].value = Rx 
        self.params['TFRy'].value = Ry
        
    def subtract_thermal(self):
        temp = self.params['TFpeak'].value
        self.params['TFpeak'].value = 0
        self.temp_image = self.image - TF_2D(self.params,self.x,self.y)
        self.params['TFpeak'].value = temp
        
    def lock_bec(self):
        self.params['TFpeak'].value = 0.0
        self.params_vary('011111111100')
        self.params['Gpeak'].value = .01
    
    def lock_gauss(self):
        temp = self.params['Gpeak'].value
        self.params['Gpeak'].value = 0
        self.params_vary('101111001100')
        self.params['TFpeak'].value = .1
        return(temp)
        
        
###########################################
########################################
 
#class for storing objects of type fit  _obj
class fit_storage(object):
    def __init__(self,names,ind_vars):
        self.names = names
        self.ind_vars = ind_vars
        self.fits = []
        self.fit_vars = []
        self.prepare_fit_obj()
        
    def prepare_fit_obj(self):
        for i in self.names:
            self.fits.append(fit_obj(i))
            index = functools.reduce(lambda x,y:x+y,[s for s in i.rsplit('_')[0] if s.isdigit()])
            self.fit_vars.append(self.grab_vals(int(index)))
            
    def do_fitting(self, show = False):
        tot = len(self.fits)
        k = 0
        for i in self.fits:
            print('Fitting number',k, 'out of' ,tot)
            try:
                i.prepare_fit()
                #fit one, get BEC RADIUS ESTIMATE
                i.fit_enhanced_bimod(i.image)
                #fit two, FIT THERMAL WINGS
                i.subtract_bec(1)
                i.lock_bec()
                i.fit_enhanced_bimod(i.temp_image)
                    #FIT THREE, FIT BEC
                i.subtract_thermal()
                temp = i.lock_gauss()
                i.fit_enhanced_bimod(i.temp_image)
                    #show results
                i.params['Gpeak'].value = temp
                if show == True:
                    i.show_fit_results()
                i.clear_images()
                i.num_atoms()
            except:
                print('Error Encountered in',i.name)
            k = k + 1
            
    def do_bimodal_fitting(self,show = False):
        for i in self.fits:
            i.prepare_fit()
            i.params_vary('111111111110')
            i.params['Gpeak'].value = .01
            i.fit_enhanced_bimod(i.image)
            if show == True:
                i.show_fit_results()

    def grab_vals(self,index):
        answer = []
        for i in self.ind_vars:
            try:
                answer.append([i[0],i[1][index]])
            except:
                answer.append([i[0],'nan'])
        return(dict(answer))
        
    def fit_out(self,filename):
        outname = filename + '_results.txt'
        output = []
        for i in range(len(self.fits)):
            if self.fits[i].results != None:
                output.append([self.fit_vars[i],self.fits[i].readout()])
        
        var_form = make_format_string(len(output[0][0]),'name')
        out_form = make_format_string(len(output[0][1]))
        out_form1 = make_format_string(len(output[0][1]),'name')                               
        var_keys = [i for i in output[0][0]]
        out_keys = [i for i in output[0][1]]
        f = open(outname,'w')
        f.write(var_form.format(*tuple(var_keys)))
        f.write(out_form1.format(*tuple(out_keys)))
        f.write('\n')
        for i in output:
            temp = [i[0][j] for j in var_keys]
            temp1 = [i[1][j] for j in out_keys]
            f.write(var_form.format(*tuple(temp)))
            f.write(out_form.format(*tuple(temp1)))
            f.write('\n')
        f.close()
        
            
#some usefull formatting things
def make_format_string(num,tf = None):
    if tf == 'name':
        form = '{:<14}'
    else:
        form = '{:<14.4f}'
    out = ''
    for i in range(num):
        out = out + form
    return(out)
        
        
#function to set up fit (there is file from igor with mat to be fit)
#pass full filepath
def set_up_fit(filename):
    var_names = ['micropow','spinortime','rf','bfield']
    f = open(filename,'r')
    data_names =  f.readlines()
    for i in range(len(data_names)):
        data_names[i] = data_names[i].rstrip('\n')
    f.close()
    ind_vars = []
    for j in var_names:
        for i in range(10):
            try:
                var = np.loadtxt(j+str(i))
                ind_vars.append([j+str(i),var])
            except:
                pass
                
    return(data_names,ind_vars)

message = "Make sure you are in the folder with data images, independant variables and input file"  
if __name__  == '__main__':
    start = time.time()
    os. getcwd()
    #path = 'C:\\Users\\zag\\Documents\\TESTING\\fitinfo.txt'
    #filepath, filename = os.path.split(path)
    #os.chdir(filepath)
    print("This is BECFitter V1.0")
    print(message)
    filename = input("Input File Name: ")
    data_names,ind_vars = set_up_fit(filename)
    all_fits = fit_storage(data_names,ind_vars)
    mid = time.time()
    all_fits.do_fitting(False)
    #all_fits.do_bimodal_fitting(True)
    mid1 = time.time()
    all_fits.fit_out(filename.rstrip('.txt'))
    end = time.time()
    print('setup',mid - start)
    print('Fit',len(all_fits.names), 'in',mid1 - mid)
    print('printing',end-mid1)
    plt.show()
    

#data = fit_obj('matrix1966_0.mat')