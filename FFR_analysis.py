# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 11:56:29 2015

@author: lwang
"""

import mne
import numpy as np
import scipy
# from anlffr import spectral
# from scipy import io
import os
import fnmatch
import pylab as pl
from scipy import io

from anlffr.helper import biosemi2mne as bs
from anlffr.preproc import find_blinks
from anlffr import spectral
from anlffr import bootstrap
# from scipy.signal import detrend
from mne.preprocessing.ssp import compute_proj_epochs
from mne.viz.topomap import _prepare_topo_plot as ptopo
#from mne.time_frequency.tfr import _induced_power as induced_power
from mne.io import edf, set_eeg_reference, make_eeg_average_ref_proj

froot = 'D:/EEG_Data/Ex012 (STHL Algorithm)/'
result_path = 'C:/Users/lwang/My Work/EEG_Experiments/Results/Ex012 (STHL Algorithm)/' 

layout=mne.channels.read_layout(kind='biosemi32.lay',
     path='C:/Users/lwang/My Work/EEG_Experiments/Analysis_Codes/Python/Ex010 (ACEMMN)/')
     
SNRs = ['no noise', '40dB', '30dB', '20dB', '10dB', '5dB', '0dB']  

subj = 'C003'
     
bdfs = fnmatch.filter(os.listdir(froot), '*' + 'STHL_' + '*MBclicks' + '*bpnoise*' +  subj + '.bdf')
fname = bdfs[0][:-4]

raw = mne.io.read_raw_edf(froot + fname + '.bdf', preload=True)   

Fs = raw.info['sfreq']
 
eves_raw = mne.find_events(raw, min_duration=2/Fs)

eves = eves_raw.copy()

eves[:,1] = np.bitwise_and(eves_raw[:,1], 255)
eves[:,2] = np.bitwise_and(eves_raw[:,2], 255)

if fname == 'Le_20160603_STHL_MBclicks_300ms_nonoise_70dB_C096':
    tmp = np.delete(eves, np.s_[18249:18447], axis=0)
    eves = tmp.copy()
    raw.info['bads'] = ['A25']
    
if eves.shape[0] == 601:
    trig1 = eves[1::3,2].copy()
    trig2 = eves[2::3,2].copy()
    trig3 = eves[3::3,2].copy()    
    true_eves = eves[3::3,:].copy()
else:
    trig1 = eves[1::4,2].copy()
    trig2 = eves[2::4,2].copy()
    trig3 = eves[3::4,2].copy()
    trig4 = eves[4::4,2].copy()
    true_eves = eves[4::4,:].copy()
    
if len(eves) == 36001:
    true_eves[:,2] = trig1*10 + trig2 + trig4*100
elif np.logical_or(len(eves) == 36000, fname=='Le_20151216_STHL_MBclicks_300ms_bpnoise_70dB_S000'):
    true_eves[:,2] = trig1[:-1]*10 + trig2[:-1] + trig4*100
elif np.mod(len(eves)-1,4) == 1:
    true_eves[:,2] = trig1[:-1]*10 + trig2 + trig4*100
elif np.mod(len(eves)-1,4) == 3:
    true_eves[:,2] = trig1[:-1]*10 + trig2[:-1] + trig4*100    
else:
    print 'Trigger number is wrong!!!!'    
    true_eves[:,2] = trig1*10 + trig2 + trig4*100

if fname == 'Le_20160127_STHL_MBclicks_300ms_bpnoise_70dB_C012':
    badCM_win = np.array([[12729500, 14561000],
                 [16881000, 17342000],
                 [21477700, 22908700]])
    
    for i in range(badCM_win.shape[0]):
        if i == 0:
            badeve_ind = np.logical_and(true_eves[:,0]<badCM_win[i,1], true_eves[:,0]>badCM_win[i,0])
        else:
            badeve_ind = np.logical_or(badeve_ind,np.logical_and(true_eves[:,0]<badCM_win[i,1], true_eves[:,0]>badCM_win[i,0]))
    true_eves = true_eves[~badeve_ind,:]
    
true_eves_bk = true_eves.copy()


#ntrials = trig2.shape[0]
#
#for i in np.arange(0, true_eves_bk.shape[0], 1):
#    if i<ntrials-1:
#        for j in np.arange(0,19,1):
#            true_eves = np.insert(true_eves, i*20+1, [true_eves_bk[i,0] + (j+1)*4096*2, 0, true_eves_bk[i,2]], axis=0)
#            
##            true_eves[i*20+1,0] = true_eves[i*20,0] + (j+1)*4096
#    elif i == ntrials-1:
#        for j in np.arange(0,19,1):  
#            true_eves = np.append(true_eves, np.asarray([true_eves_bk[-1,0] + (j+1)*4096*2, 0, true_eves_bk[-1,2]]).reshape(1,3),axis=0)        
#            
#true_eves = true_eves[true_eves[:,0].argsort()]


refChannels = ['EXG1', 'EXG2']

print 'Re-referencing data to', refChannels
(raw, ref_data) = set_eeg_reference(raw, refChannels, copy=False)    

raw.filter(l_freq=70, h_freq=1500, picks=np.arange(0, 32, 1), n_jobs=8)
 
if subj == 'C015':
    raw.info['bads'] = ['A20']
#elif subj == 'C012':
#    raw.info['bads'] = ['A3','A6','A7','A23','A24', 'A25','A27', 'A28']    
        
if subj == 'S007':    
    raw.info['bads'] = ['A17', 'A18'] # for S007

trig_list = np.unique(true_eves_bk[:,2])
n_trigs = trig_list.shape[0]

conds = {}
for i in range(n_trigs):                
        conds['trigger' + str(trig_list[i])] = trig_list[i]
        
epochs = mne.Epochs(raw, true_eves, conds, tmin=0, proj=False,
                    tmax=0.3, baseline=(0, 0.3), picks=np.arange(0, 32, 1),
                    reject=dict(eeg=200e-6),preload=True)
epochs, indices = epochs.equalize_event_counts(conds,method='truncate')

x = {}
for icond, cond in enumerate(conds):    
    
    data = epochs[cond].get_data() #[:,ch,:].mean(axis=1, keepdims=True)
    data = data[:,0:32,:].transpose((1,0,2))
    x[cond] = data
    
#print x.keys()
#print conds
    
nPerDraw = 800
nDraws = 20
params = {'Fs': Fs, 'tapers': [1,1], 'fpass': [70, 1500], 'itc': 1}
ncond = np.unique(eves[:,2])[:-3].max()

for j in range(ncond):        
    data_hp_pos = x['trigger101' + str(j+1)]
    data_hp_neg = x['trigger111' + str(j+1)]
    data_hp = np.concatenate((data_hp_pos,data_hp_neg), axis=1)
     
    
    print 'Running CPCA Spec/PLV Estimation for condition '+str(j)
    params = spectral.generate_parameters(Fs = Fs,
                                          tapers = [1,1],
                                          fpass = [70,1500],
                                          itc = 1,
                                          threads=4, 
                                          nPerDraw=nPerDraw, 
                                          nDraws=nDraws)
    datalist = [data_hp_pos, data_hp_neg]
    results = bootstrap.bootfunc(spectral._mtcpca_complete, datalist, params)   
    f = results['f']
    mplv_SN = results['mtcpcaPLV_normalPhase']['bootMean']
    vplv_SN = results['mtcpcaPLV_normalPhase']['bootVariance']
    mplv_N = results['mtcpcaPLV_noiseFloorViaPhaseFlip']['bootMean']
    vplv_N = results['mtcpcaPLV_noiseFloorViaPhaseFlip']['bootVariance']
    
    mspec_SN = results['mtcpcaSpectrum_normalPhase']['bootMean']
    vspec_SN = results['mtcpcaSpectrum_normalPhase']['bootVariance']
    mspec_N = results['mtcpcaSpectrum_noiseFloorViaPhaseFlip']['bootMean']
    vspec_N = results['mtcpcaSpectrum_noiseFloorViaPhaseFlip']['bootVariance']       
    
#        print 'Running CZ channel plv estimation'
#        (mplv,vplv,f) = spectral.bootfunc(data_hp[31,:,:].squeeze(),nPerDraw,nDraws,params,func='plv')   
    
    
        
    if j == 0:
        mplv_SN_allcond = np.zeros((ncond,mplv_SN.size))      
        vplv_SN_allcond = np.zeros((ncond,vplv_SN.size))
        mspec_SN_allcond = np.zeros((ncond,mspec_SN.size))      
        vspec_SN_allcond = np.zeros((ncond,vspec_SN.size))
        mplv_N_allcond = np.zeros((ncond,mplv_N.size))      
        vplv_N_allcond = np.zeros((ncond,vplv_N.size))
        mspec_N_allcond = np.zeros((ncond,mspec_N.size))      
        vspec_N_allcond = np.zeros((ncond,vspec_N.size))
        
    mplv_SN_allcond[j,:] = mplv_SN
    vplv_SN_allcond[j,:] = vplv_SN
    mplv_N_allcond[j,:] = mplv_N
    vplv_N_allcond[j,:] = vplv_N
    mspec_SN_allcond[j,:] = mspec_SN
    vspec_SN_allcond[j,:] = vspec_SN
    mspec_N_allcond[j,:] = mspec_N
    vspec_N_allcond[j,:] = vspec_N
        
        
    
 # Saving Results
res = dict(SNRs=SNRs, mplv_SN=mplv_SN_allcond, 
                      vplv_SN=vplv_SN_allcond,
                      mplv_N=mplv_N_allcond, 
                      vplv_N=vplv_N_allcond,
                      mspec_SN=mspec_SN_allcond, 
                      vspec_SN=vspec_SN_allcond,
                      mspec_N=mspec_N_allcond, 
                      vspec_N=vspec_N_allcond,
                      f=f, plv_params=params, Fs=Fs)
respath = result_path + subj

save_name = fname + '_cpcaENV_ALL.mat'
del params['returnIndividualBootstrapResults']
    
if (not os.path.isdir(respath)):
    os.mkdir(respath)

io.savemat(respath + '/' + save_name, res)
    
## Phase analysis for CZ    
for i in range(1):
    for j in range(ncond):        
        data_hp_pos = x['trigger10' + str(10*(i+1)+j+1)]
        data_hp_neg = x['trigger11' + str(10*(i+1)+j+1)]
        data_hp = np.concatenate((data_hp_pos,data_hp_neg), axis=1)

        data_cz = data_hp[31,:,:]
        Data_cz = scipy.fftpack.fft(data_cz)
        ph_cz = np.angle(Data_cz)

        if np.logical_and(i==0, j==0):
            ph_cz_allcond = np.zeros((ncond,ph_cz.shape[0], ph_cz.shape[1]))

        ph_cz_allcond[j,:,:] = ph_cz

freq = np.arange(0,Fs,Fs/ph_cz_allcond.shape[-1])
res = dict(freq=freq, ph_cz_allcond=ph_cz_allcond, Fs=Fs)
#res = dict(SNRs=SNRs, S_hp=S_allcond_hp, N_hp=N_allcond_hp, vS_hp=vS_allcond_hp, vN_hp=vN_allcond_hp,
#           f=f, plv_params=params, Fs=Fs, nPerDraw=nPerDraw, nDraws=nDraws)
respath = result_path + subj
save_name = fname + '_phaseCZ.mat'
if (not os.path.isdir(respath)):
    os.mkdir(respath)

io.savemat(respath + '/' + save_name, res)
    
    
#plot for multiband clicks in quiet
#pl.figure()
#for i in range(1):
#    for j in range(1):
#        ax=pl.subplot(1,6,1)
#        pl.plot(f, mplv_allcond_hp[0,:,:].T)
#        pl.xlim([104,124])
#        pl.subplot(1,6,2)
#        pl.plot(f, mplv_allcond_hp[0,:,:].T)
#        pl.xlim([160,180])
#        pl.subplot(1,6,3)
#        pl.plot(f, mplv_allcond_hp[0,:,:].T)
#        pl.xlim([226,246])
#        pl.subplot(1,6,4)
#        pl.plot(f, mplv_allcond_hp[0,:,:].T)
#        pl.xlim([218,238])
#        pl.subplot(1,6,5)
#        pl.plot(f, mplv_allcond_hp[0,:,:].T)
#        pl.xlim([330,350])
#        pl.subplot(1,6,6)
#        pl.plot(f, mplv_allcond_hp[0,:,:].T)
#        pl.xlim([462,482])
#        
#        if np.logical_and(i == 0, j == 0):
#            pl.legend(['no noise','low','mid', 'high','low+mid','low+high','mid+high','all bands','frozen LBnoise'])
#if analysis_type == 'PLV':
#    pl.suptitle(subj+'MBclicks 70dB noise PLVenv')
#elif analysis_type == 'Spec':
#    pl.suptitle(subj+'MBclicks 70dB noise Specenv')
#
#pl.figure()
#F0=np.asarray([114,170,236])
#f_ind = np.zeros((3,5))
#for i in range(3):
#    for j in range(5):
#        pl.subplot(3,5,5*i+j+1)
#        freq = (j+1)*F0[i]
#        f_ind[i,j] = np.abs(f-freq).argmin()
#        pl.errorbar(np.arange(1,ncond+1), mplv_allcond_hp[0,:,f_ind[i,j]], np.sqrt(vplv_allcond_hp[0,:,np.abs(f-freq).argmin()]))
#        pl.xlim((0,ncond+1))
#        pl.xticks(np.arange(1,ncond+1), ['noN','L', 'M', 'H','LM','LH','MH','LMH','LB_N'])
#        pl.title('frequency = '+str(freq)+'Hz')
##        pl.ylim([0,0.2])
#if analysis_type == 'PLV':
#    pl.suptitle('mean cplvENV at harmonics of MBclicks')
#elif analysis_type == 'Spec':
#    pl.suptitle('mean cspecENV at harmonics of MBclicks')    
#    
#if analysis_type == 'Spec':    
#    pl.figure()
#    f_ind = f_ind.astype(int)
#    for i in range(3):
#        pl.subplot(3,1,i+1)    
#        mplv_sum = mplv_allcond_hp[0,:,f_ind[i,:]].sum(axis=0)
#        vplv_sum = vplv_allcond_hp[0,:,f_ind[i,:]].sum(axis=0)   
#        pl.errorbar(np.arange(1,ncond+1), mplv_sum, np.sqrt(vplv_sum))
#        pl.xlim((0,ncond+1))
#        pl.xticks(np.arange(1,ncond+1), ['noN','L', 'M', 'H','LM','LH','MH','LMH','LB_N'])
#        pl.title('F0 = '+str(F0[i])+'Hz')
#    pl.suptitle('sum of cspecENV at harmonics 1-5 of MBclicks'+' ('+subj+')',fontsize=20)         

#freq = np.arange(0,Fs,Fs/ph_cz_allcond.shape[-1])
#plv_cz_allcond = np.abs(np.exp(1j*ph_cz_allcond).mean(axis=1))
#pl.figure()
#pl.plot(freq, plv_cz_allcond.T)
#pl.xlim([100,300])
#
#conds = ['no noise','L', 'M', 'H', 'LM', 'LH', 'MH', 'LMH','frozen LBnoise']
#for j in range(F0.shape[0]):
#    pl.figure()
#    for i in range(ncond):
#        pl.subplot(2,5,i+1)
#        [hist, bin_edges]=np.histogram(ph_cz_allcond[i,:,np.abs(freq-F0[j]).argmin()],bins=50, range=(-np.pi, np.pi))
#        pl.bar(bin_edges[:-1],hist,width=0.1)
#        pl.title(conds[i])
#    pl.suptitle('phase distribution of CZ at '+str(F0[j])+'Hz ('+subj+')',fontsize=20)
#
##meanph_Lb = np.angle(np.mean(np.exp(1j*ph_cz_allcond[np.asarray([0,3,4,6]),:,np.abs(freq-F0[0]).argmin()]),axis=1))
##meanph_Mb = np.angle(np.mean(np.exp(1j*ph_cz_allcond[np.asarray([1,3,5,6]),:,np.abs(freq-F0[1]).argmin()]),axis=1))
##meanph_Hb = np.angle(np.mean(np.exp(1j*ph_cz_allcond[np.asarray([2,4,5,6]),:,np.abs(freq-F0[2]).argmin()]),axis=1))
#
#meanph_Lb = np.angle(np.mean(np.exp(1j*ph_cz_allcond[np.asarray([0,2,3,6]),:,np.abs(freq-F0[0]).argmin()]),axis=1))
#meanph_Mb = np.angle(np.mean(np.exp(1j*ph_cz_allcond[np.asarray([0,1,3,5]),:,np.abs(freq-F0[1]).argmin()]),axis=1))
#meanph_Hb = np.angle(np.mean(np.exp(1j*ph_cz_allcond[np.asarray([0,1,2,4]),:,np.abs(freq-F0[2]).argmin()]),axis=1))
#meanph_Mb[meanph_Mb>0] = meanph_Mb[meanph_Mb>0]-2*np.pi
#
#
#pl.figure()
#pl.scatter(np.ones((4,))*F0[0],meanph_Lb/(2*np.pi))
#pl.scatter(np.ones((4,))*F0[1],meanph_Mb/(2*np.pi))
#pl.scatter(np.ones((4,))*F0[2],meanph_Hb/(2*np.pi)-1)
#pl.xlabel('Mod F (Hz)')
#pl.ylabel('mean phase (cycles)')
#pl.title('S000 FFR phases at CZ')
#delay1 = (np.mean(meanph_Mb/(2*np.pi))-np.mean(meanph_Lb/(2*np.pi)))/(F0[1]-F0[0])
#delay2 = (np.mean(meanph_Hb/(2*np.pi)-1)-np.mean(meanph_Mb/(2*np.pi)))/(F0[2]-F0[1])
#pl.text(140, -0.10, 'delay='+str(round(delay1*10000)/10)+'ms',color='r')
#pl.text(200, -0.50, 'delay='+str(round(delay2*10000)/10)+'ms',color='r')
#pl.plot(F0,[np.mean(meanph_Lb/(2*np.pi)), np.mean(meanph_Mb/(2*np.pi)), np.mean(meanph_Hb/(2*np.pi)-1)],'r')
#y=np.asarray([meanph_Lb/(2*np.pi), meanph_Mb/(2*np.pi), meanph_Hb/(2*np.pi)-1])
#x=F0
#fit = np.polyfit(x,np.mean(y,axis=1),1)
#yfit=fit[0]*np.asarray(x)+fit[1]
#pl.plot(x,yfit,'g')
#pl.text(140, -0.60, 'delay='+str(round(fit[0]*10000)/10)+'ms',color='g')

    
# plot the PLV for 114Hz BPclicks at different levels
#pl.figure()
#for i in range(2):
#    for j in range(5):
#        ax=pl.subplot(3,5,5*i+j+1)
#        pl.plot(f, mplv_allcond_hp[:,i,:].T)
#        pl.xlim([114*j+104,114*(j+1)+10])
#        
#        if np.logical_and(i == 2, j == 4):
#            pl.legend(['55dB','65dB', '75dB'])
#        else:
#            pl.text(0.6,0.8,str((j+1)*114) + 'Hz',transform=ax.transAxes)
#pl.suptitle('S000 BPclicks114Hz varystimlevel cPLVenv')
#
#stimlevel = ['55dB', '65dB', '75dB']
#pl.figure()
#for i in range(3):
#    pl.subplot(3,1,i+1)
#    pl.errorbar(np.arange(1,4), mplv_allcond_hp[i,:,22], np.sqrt(vplv_allcond_hp[i,:,22]))
#    pl.xlim((0,4))
#    pl.xticks(np.arange(1,4), ['no noise', 'frozen noise', 'unfrozen noise'])
#    pl.title('stim level = '+stimlevel[i])
#pl.suptitle('mean cplvENV at F0 (114Hz)')
    
    
### For 500Hz, 4000Hz FS PLV plotting    
#pl.figure()
#for i in range(2):    
#    
#    if i==0:
#        ax=pl.subplot(2,2,i+1)
#        pl.plot(f, mplv_allcond_hp[:,i,:].T)
#        pl.xlim([450,550])
#        pl.title('CF = 500 Hz')
#        ax=pl.subplot(2,2,i+2)
#        pl.plot(f, mplv_allcond_hp[:,i,:].T)
#        pl.xlim([3950,4050])        
#        pl.title('CF = 500 Hz')
#    else:
#        ax=pl.subplot(2,2,i+2)
#        pl.plot(f, mplv_allcond_hp[:,i,:].T)
#        pl.xlim([450,550])
#        pl.title('CF = 4000 Hz')
#        ax=pl.subplot(2,2,i+3)
#        pl.plot(f, mplv_allcond_hp[:,i,:].T)
#        pl.xlim([3950,4050])        
#        pl.title('CF = 4000 Hz')
#        pl.legend(['55dB', '75dB'])
#pl.suptitle('S000 puretone varystimlevel cPLVenv')   
#
#CFs = ['500 Hz', '4000 Hz']
#pl.figure()
#for i in range(2):
#    pl.subplot(2,1,i+1)
#    if i==0:
#        pl.errorbar(np.arange(1,3), mplv_allcond_hp[:,i,215], np.sqrt(vplv_allcond_hp[:,i,215]))
#    else:
#        pl.errorbar(np.arange(1,3), mplv_allcond_hp[:,i,1965], np.sqrt(vplv_allcond_hp[:,i,1965]))
#        
#    pl.xlim((0,3))
#    pl.xticks(np.arange(1,3), ['55dB', '75dB'])
#    pl.title('Pure tone = '+CFs[i])
#pl.suptitle('mean cplvENV at CF')     
#    
#
#for i in range(2):
#    pl.figure()
#    
#    for j in range(2):
#        pl.subplot(2,2,j*2+1)        
#        tp,cn=mne.viz.topomap.plot_topomap(plv_allcond_hp[i,j,:,np.abs(f-500).argmin()].squeeze(), layout.pos[:,:2],vmin=0,vmax=0.09)
#        if j==0:
#            pl.title('CF = 500Hz @500Hz')
#        else:
#            pl.title('CF = 4000Hz @500Hz')
#        pl.subplot(2,2,j*2+2)        
#        tp,cn=mne.viz.topomap.plot_topomap(plv_allcond_hp[i,j,:,np.abs(f-4000).argmin()].squeeze(), layout.pos[:,:2],vmin=0,vmax=0.09)
#        if j==0:
#            pl.title('CF = 500Hz @4000Hz')
#        else:
#            pl.title('CF = 4000Hz @4000Hz')
#
#    if i==0:
#        pl.suptitle('Level = 55dB')
#    else:
#        pl.suptitle('Level = 75dB')
