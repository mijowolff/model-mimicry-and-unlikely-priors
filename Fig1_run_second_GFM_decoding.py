#%% 
# This script produces the results shown in Figure 1 in Wolff and Rademaker (2024)
# It loads the simulation data obtained from Fig1_run_first_GFM_simulations.m and performs the decoding analysis
# precomputed simulation data and decoding results are available at https://osf.io/bdf74/

import scipy.io as sio
import numpy as np
from custom_functions import dist_theta_kfold,circ_dist,circ_mean
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import pickle
import seaborn as sns 
import pandas as pd 
from scipy.stats import circstd

dat_dir='' # path to simulation data
results_dir='' # path to results 

fig_dir_out=''
save_fig=True # save figures as .svg
do_decoding=True # run decoding analysis?

dat_all=sio.loadmat(dat_dir+'/GFM_simulation_data.mat') # load simulation data
#
# extract everything from dat_all
theta_deg_all=dat_all['stimuli_subs']
data_gain_all=dat_all['sensor_response_gain_subs']
data_width_all=dat_all['sensor_response_width_subs']
data_pref_all=dat_all['sensor_response_pref_subs']
data_uni_all=dat_all['sensor_response_uni_subs']

tcurves_pref=dat_all['tcurves_pref']
tcurves_gain=dat_all['tcurves_gain']
tcurves_width=dat_all['tcurves_width']
tcurves_uni=dat_all['tcurves_uni']
signal_levels=dat_all['signal_levels']

thetas=[]

dec_pref=[]
dec_gain=[]
dec_width=[]
dec_uni=[]
dists_pref=[]
dists_gain=[]
dists_width=[]
dists_uni=[]

ang_steps=8# number of times to shift the orientations bins when decoding
n_reps=20 # number of repetitions for the decoder (with random folds each time)
n_folds=8 # number of folds 

nsubs=theta_deg_all.shape[0]
#%%
if do_decoding:
    for sub in range(nsubs):
        print(sub+1)
        
        theta_deg=theta_deg_all[sub,:]
        theta_deg[theta_deg<0]=theta_deg[theta_deg<0]+180
        
        ntrls=theta_deg.shape[0]
      
        theta=np.deg2rad(theta_deg)*2
        thetas.append(theta)

        # preferred tuning model
        dat_temp=data_pref_all[sub,:,:]
        dec_cos,_,distances_ordered,_,angspace_full=dist_theta_kfold(dat_temp,theta,n_reps=n_reps,n_folds=n_folds,ang_steps=ang_steps)
        dec_pref.append(dec_cos)
        dists_pref.append(distances_ordered)

        # gain model
        dat_temp=data_gain_all[sub,:,:]
        dec_cos,_,distances_ordered,_,angspace_full=dist_theta_kfold(dat_temp,theta,n_reps=n_reps,n_folds=n_folds,ang_steps=ang_steps)
        dec_gain.append(dec_cos)
        dists_gain.append(distances_ordered)

        # width model
        dat_temp=data_width_all[sub,:,:]
        dec_cos,_,distances_ordered,_,angspace_full=dist_theta_kfold(dat_temp,theta,n_reps=n_reps,n_folds=n_folds,ang_steps=ang_steps)
        dec_width.append(dec_cos)
        dists_width.append(distances_ordered)

        # SNR model
        dat_temp=data_uni_all[sub,:,:]
        dec_cos,_,distances_ordered,_,angspace_full=dist_theta_kfold(dat_temp,theta,n_reps=n_reps,n_folds=n_folds,ang_steps=ang_steps)
        dec_uni.append(dec_cos)
        dists_uni.append(distances_ordered)

    with open(results_dir+'/GFM_simulations_decoding.pickle','wb') as f:
        pickle.dump([dec_pref,dists_pref,dec_gain,dists_gain,dec_width,dists_width,dec_uni,dists_uni,tcurves_pref,tcurves_gain,tcurves_width,tcurves_uni,signal_levels,thetas,angspace_full],f)
               
#%%
with open(results_dir+'/GFM_simulations_decoding.pickle','rb') as f:
    dec_pref,dists_pref,dec_gain,dists_gain,dec_width,dists_width,dec_uni,dists_uni,tcurves_pref,tcurves_gain,tcurves_width,tcurves_uni,signal_levels,thetas,angspace_ful=pickle.load(f)       

# make orientation bins, the bin witdh is 22.5 and we have 16 time 8 bins in total (so there is overlap between bins!)
ang_bins_temp=np.arange(0,2*np.pi,np.pi/8)
bin_width=np.diff(ang_bins_temp)[0]
ang_bins=np.zeros((8,len(ang_bins_temp)))
for i in range(8):
    ang_bins[i,:]=ang_bins_temp+i*bin_width/8

ang_bins_deg=np.round(np.rad2deg(np.reshape(ang_bins,(ang_bins.shape[0]*ang_bins.shape[1]),order='F')),15)
for isc in range(4): # loob over each simulation condition
    if isc==0:
        orient_4d_dec=dec_pref
        orient_4d_dists=dists_pref
        tcurves=tcurves_pref
    elif isc==1:
        orient_4d_dec=dec_width
        orient_4d_dists=dists_width
        tcurves=tcurves_width
    elif isc==2:
        orient_4d_dec=dec_gain
        orient_4d_dists=dists_gain
        tcurves=tcurves_gain
    elif isc==3:
        orient_4d_dec=dec_uni
        orient_4d_dists=dists_uni
        tcurves=tcurves_uni

    nsubs=len(orient_4d_dec)
    dec_bins=np.zeros((nsubs,ang_bins.shape[0],ang_bins.shape[1]))
    prec_bins=np.zeros((nsubs,ang_bins.shape[0],ang_bins.shape[1]))
    dir_bins=np.zeros((nsubs,ang_bins.shape[0],ang_bins.shape[1]))

    for isub in range(nsubs):
        dec_temp=orient_4d_dec[isub]
        dists_temp=orient_4d_dists[isub]
        theta_temp=thetas[isub]

        for abin in range(ang_bins.shape[0]):
            
            ang_bin_temp=ang_bins[abin,:]

            # bin locations into current bin space
            temp=np.argmin(abs(circ_dist(ang_bin_temp,theta_temp,all_pairs=True)),axis=1)
            ang_bin=np.tile(ang_bin_temp,(len(theta_temp),1))        
            theta_bins=ang_bin[:,temp][0,:]

            for ibin,bin in enumerate(ang_bin_temp):
                dists_bin_temp=np.squeeze(-dists_temp[:,theta_bins==bin,:])
                dists_bin_temp_av=-np.mean(dists_temp[:,theta_bins==bin,:],axis=1)
                dir_bins[isub,abin,ibin]=np.rad2deg(circ_mean(angspace_full,w=dists_bin_temp_av[:,0]))/2 
                dec_bins[isub,abin,ibin]=np.mean(dec_temp[theta_bins==bin])

                dirs_bin_temp=np.zeros((dists_bin_temp.shape[1]))
                for bin_trl in range(dists_bin_temp.shape[1]):
                    dirs_bin_temp[bin_trl]=(circ_mean(angspace_full,w=dists_bin_temp[:,bin_trl]))
                prec_bins[isub,abin,ibin]=-(circstd(dirs_bin_temp,high=np.pi,low=-np.pi))/2
    
    dec_bins=np.reshape(dec_bins,(dec_bins.shape[0],dec_bins.shape[1]*dec_bins.shape[2]),order='F')
    dec_bins_mc=dec_bins-np.mean(dec_bins,axis=1,keepdims=True) # convert to relative decoding accuracy by mean centering
    dir_bins=np.reshape(dir_bins,(dir_bins.shape[0],dir_bins.shape[1]*dir_bins.shape[2]),order='F')
    prec_bins=np.copy(np.reshape(prec_bins,(prec_bins.shape[0],prec_bins.shape[1]*prec_bins.shape[2]),order='F'))
    prec_bins_mc=prec_bins-np.mean(prec_bins,axis=1,keepdims=True) # convert to relative precision by mean centering
    
    # smooth across bins
    sigma=2/(180/len(angspace_full))
    dec_bins_mc_smooth=np.concatenate((dec_bins_mc,dec_bins_mc,dec_bins_mc),axis=1)
    dec_bins_mc_smooth=gaussian_filter1d(dec_bins_mc_smooth,sigma=sigma,axis=1)
    dec_bins_mc_smooth=dec_bins_mc_smooth[:,dec_bins_mc.shape[1]:2*dec_bins_mc.shape[1]]

    dir_bins_smooth=np.concatenate((dir_bins,dir_bins,dir_bins),axis=1)
    dir_bins_smooth=gaussian_filter1d(dir_bins_smooth,sigma=sigma,axis=1)
    dir_bins_smooth=dir_bins_smooth[:,dir_bins.shape[1]:2*dir_bins.shape[1]]

    prec_bins_mc_smooth=np.concatenate((prec_bins_mc,prec_bins_mc,prec_bins_mc),axis=1)
    prec_bins_mc_smooth=gaussian_filter1d(prec_bins_mc_smooth,sigma=sigma,axis=1)
    prec_bins_mc_smooth=prec_bins_mc_smooth[:,prec_bins_mc.shape[1]:2*prec_bins_mc.shape[1]]
    angspace_ang=np.rad2deg(angspace_full)/2+90
   
    dec_bins_mc_smooth_df=pd.DataFrame(dec_bins_mc_smooth)
    dir_bins_smooth_df=pd.DataFrame(dir_bins_smooth)
    prec_bins_mc_smooth_df=pd.DataFrame(prec_bins_mc_smooth)
    temp=np.tile(angspace_ang,(1,nsubs)).flatten()
    # convert columns in dec_bins_mc_smooth_df to orientation bins
    dec_bins_mc_smooth_df.columns=angspace_ang
    dir_bins_smooth_df.columns=angspace_ang
    prec_bins_mc_smooth_df.columns=angspace_ang
    # add columns for subjects
    dec_bins_mc_smooth_df['subject']=np.arange(1,nsubs+1)
    dir_bins_smooth_df['subject']=np.arange(1,nsubs+1)
    prec_bins_mc_smooth_df['subject']=np.arange(1,nsubs+1)
    # convert to long format
    dec_bins_mc_smooth_df=pd.melt(dec_bins_mc_smooth_df,id_vars=['subject'],var_name='orientation',value_name='decoding accuracy')
    dir_bins_smooth_df=pd.melt(dir_bins_smooth_df,id_vars=['subject'],var_name='orientation',value_name='direction')
    prec_bins_mc_smooth_df=pd.melt(prec_bins_mc_smooth_df,id_vars=['subject'],var_name='orientation',value_name='precision')

    #%
    dec_acc_ylim=[-0.009,0.009]
    prec_ylim=[-0.16,0.16]
    dir_ylim=[-30,30]
    dir_diff_ylim=[-20,20]

    angspace_ang180=np.arange(1,181,1)
    n_boot=1000
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(3, 9),
                        gridspec_kw={
                            'height_ratios': [1,1,1,1],
                        'wspace': .6,
                        'hspace': .1})
    if isc==0:
        ax[0].set_title('Preferred tuning simulations')
    elif isc==1:
        ax[0].set_title('Width tuning simulations')
    elif isc==2:
        ax[0].set_title('Gain tuning simulations')
    elif isc==3:
        ax[0].set_title('SNR simulations')

    plt.axes(ax[0])
    plt.plot(angspace_ang180,tcurves)
    plt.xticks(np.arange(0,181,22.5),[])
    plt.ylabel('Channel response')
    plt.xlim([0,180])
    if isc==0:
        plt.ylim([0,1.1])
    elif isc==1:
        plt.ylim([0,1.6])
    elif isc==2:
        plt.ylim([0,1.6])
    elif isc==3:
        plt.ylim([0,1.1])

    if isc==3:
        # make second y axis for SNR model
        ax2 = ax[0].twinx()
        ax2.set_ylabel('SNR', color='r')
        ax2.plot(angspace_ang180,signal_levels,color='r',linewidth=3)

    plt.axes(ax[1])
    sns.lineplot(x='orientation',y='decoding accuracy',data=dec_bins_mc_smooth_df,errorbar=('ci',95),n_boot=n_boot)
    # plot vertical line at 0
    plt.axvline(90,color='k',linestyle='--',linewidth=1)
    # plot horizontal line at 0
    plt.axhline(0,color='k',linestyle='--',linewidth=1)
    # plt.yticks(np.arange(-.01,.011,.005))

    plt.xlim([0,180])
    plt.ylim(dec_acc_ylim)
    plt.xticks(np.arange(0,181,22.5),[])
    plt.xlabel('')
    plt.ylabel('Relative decoding accuracy')

    plt.axes(ax[2])
    sns.lineplot(x='orientation',y='precision',data=prec_bins_mc_smooth_df,errorbar=('ci',95),n_boot=n_boot)
    # plot vertical line at 0
    plt.axvline(90,color='k',linestyle='--',linewidth=1)
    # plot horizontal line at 0
    plt.axhline(0,color='k',linestyle='--',linewidth=1)

    plt.xlim([0,180])
    plt.ylim(prec_ylim)
    plt.xticks(np.arange(0,181,22.5),[])
    plt.xlabel('')
    plt.ylabel('Relative precision')

    plt.axes(ax[3])
    sns.lineplot(x='orientation',y='direction',data=dir_bins_smooth_df,errorbar=('ci',95),n_boot=n_boot)
    # plot vertical line at 0
    plt.axvline(90,color='k',linestyle='--',linewidth=1)
    # plot horizontal line at 0
    plt.axhline(0,color='k',linestyle='--',linewidth=1)

    plt.xlim([0,180])
    plt.ylim(dir_ylim)
    plt.xticks(np.arange(0,181,22.5),rotation=-45)
    plt.xlabel('Orientation (deg)')
    plt.ylabel('Bias (deg)')

    if isc==0:
        fig_name='/Preferred_mahal'
    elif isc==1:
        fig_name='/Width_mahal'
    elif isc==2:
        fig_name='/Gain_mahal'
    elif isc==3:
        fig_name='/SNR_mahal'

    # save figure as .svg
    if save_fig:
        plt.savefig(fig_dir_out+fig_name+ '.svg',dpi=300,bbox_inches='tight')

    # plt.show()


        