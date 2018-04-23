# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 10:20:42 2015

@author: lwang
"""
from figure2pdf import figure2pdf
from scipy import io
import pylab as pl
import mne
import numpy as np
import os
import fnmatch

pl.close('all')

# Adding Files and locations
#froot = 'D:/EEG_Data/Ex012 (STHL Algorithm)/'
result_path = 'C:/Users/lwang/My Work/EEG_Experiments/Results/Ex012 (STHL Algorithm)/' 

layout=mne.channels.read_layout(kind='biosemi32.lay',
     path='C:\Users\lwang\My Work\EEG_Experiments\Analysis_Codes\Python\Ex010 (ACEMMN)')

subjlist = [
#    'S000', 'S003', 'S006', 'S008', 'S005', 'S001', 'S004', 'S012', 'S013', 'S014',
    'S000'
            ]

paradigm = 'STHLprofile_75dB'
SNR = 'nonoise'

for k, subj in enumerate(subjlist):
    
    # These are so that the generated files are organized better    

    print 'Loading Subject', subj
    
    mats = fnmatch.filter(os.listdir(result_path+subj), '*' + paradigm + '*' + SNR + '*' + subj + '*FFRresult.mat')
    
    if len(mats) > 1:
        print '***WARNING!! Multiple files found!'
        break
    else:
        print 'Viola! Data files found!'
        fname = mats[0][:-4] # ignore the '.bdf' at the end
        
    
    # load saved results and plot
    allvars = io.loadmat(result_path+subj+'/'+fname)

    f = allvars['f'].T
    plv_allcond = allvars['plv']
    Fs = allvars['Fs']
    cplv_allcond = allvars['cplv']
    mod_depths = allvars['mod_depths']
    params = allvars['plv_params']
    CFs = allvars['CFs']
    
    for i in range(4):
        pl.figure()
        pl.hold(True)
        pl.plot(f, cplv_allcond[i,:,:].T)
        pl.title('CF = ' + CFs[i])
        if i == 0:
            pl.legend(mod_depths)
        pl.xlim([70, 170])
        pl.ylim([0, 0.1])
        
        pl.figure()
        for j in range(5):
            pl.subplot(2,3,j)
            pl.hold(True)
            tp,cn=mne.viz.topomap.plot_topomap(plv_allcond[:,i,j,np.abs((f-100)).argmin()].squeeze(), layout.pos[:,:2],vmin=0,vmax=0.15)   
            pl.title(mod_depths[j])
            if j == 4:
                pl.colorbar(tp)
    
    
    
#    figure2pdf(np.asarray(range(1,9)), respath + '/' + fname + '_FFRresult.pdf')
    
    