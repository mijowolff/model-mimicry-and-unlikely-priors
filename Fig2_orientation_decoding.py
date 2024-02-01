#%% 
# This script produces the results shown in Figure 2 in Wolff and Rademaker (2024)
# reduced data from Wolff et al. 2015, 2017 and 2020, used in this script, and precomputed results from all datasets is available at https://osf.io/bdf74/
# data used by Harrison et al. 2023 is publicly available at https://osf.io/5ba9y/

import scipy.io as sio
import numpy as np
from custom_functions import dat_prep_4d_section,dist_theta_kfold,circ_dist,circ_mean
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import pickle
import seaborn as sns 
import pandas as pd 
import os
import mat73
from scipy.stats import circstd
from mne.stats import permutation_t_test

dat_dir='' # to data for decoding
results_dir='' # directory to save results or load results from 
fig_dir_out='' # directory to save figures to
save_fig=True # save figures as .svg

do_decoding_H=True # decode the data from Harrison et al. 2023
do_decoding_W15=True # decode the data from Wolff et al. 2015
do_decoding_W17=True # decode the data from Wolff et al. 2017
do_decoding_W20=True# decode the data from Wolff et al. 2020

# variables that are consistent across experiments
toi=np.asarray([.05, .45]) # time window of interest
ang_steps=8 # number of times to shift the orientations bins when decoding
n_reps=20 # number of repetitions for the decoder (with random folds each time)
n_folds=8 # number of folds 
#%% Decode orientation from Harrison et al. 2023
if do_decoding_H:
    # posterior channels to include 
    post_chans = [ 'P7','P5','P3','P1','Pz','P2','P4','P6','P8','PO7','PO3','POz','PO4','PO8','O2','O1','Oz','Iz','P10','P9']
    
    hz=512 # sampling rate of data
    span=((1/hz)*10)*1000 # length of each time bin (in ms) when creating 4d data (adjusted for sampling rate)
    
    files = [i for i in os.listdir(dat_dir) if os.path.isfile(os.path.join(dat_dir,i)) and \
            'eeg_recordings_subject_' in i]
    files.sort()

    orient_4d_dec=[]; orient_4d_dists=[];  thetas=[]
    for isub,file in enumerate(files):
        print(isub+1)
        
        dat=mat73.loadmat(dat_dir+file)   
        eeg_dat=dat['eeg']['data']
        theta=dat['stimulus']['orientations']
        theta=theta*2 # convert to 360 degrees space
        time=dat['eeg']['times']/1000
        eeg_dat=np.transpose(eeg_dat,(2,0,1)) 
        chanlocs=dat['eeg']['chanlocs']
        eeg_labels=[] 
        for i in range(len(chanlocs)):
            eeg_labels.append(chanlocs[i]['labels'])

        # combine channel and temporal dimensions
        dat_4d=dat_prep_4d_section(eeg_dat[:,np.in1d(eeg_labels, post_chans),:],time_dat=time,toi=toi,span=span,hz=hz,relative_baseline=True,in_ms=True)

        # change orientation labels so that 0 degrees is horizontal
        theta=theta+np.pi

        dec_acc,_,distances_ordered,_,angspace_full=dist_theta_kfold(dat_4d,theta,n_reps=n_reps,n_folds=n_folds,ang_steps=ang_steps)

        orient_4d_dec.append(dec_acc) 
        orient_4d_dists.append(distances_ordered)
        thetas.append(theta)

    with open(results_dir+'/Orient_dec_4d_Harrison23.pickle','wb') as f:
        pickle.dump([orient_4d_dec,orient_4d_dists,thetas,angspace_full],f)

#%% Decode orientation from Wolff et al. 2015
if do_decoding_W15:
    hz=500
    span=20
    orient_4d_dec=[]; orient_4d_dists=[];  thetas=[]
    for isub in range(24):
        print(isub+1)
        dat=sio.loadmat(dat_dir+'/dat_2015_mem_item_' +str(isub+1)+'.mat')    
        eeg_dat1=dat['ft_mem_long'][0,0]['trial']
        theta1=dat['ft_mem_long'][0,0]['theta_rad']*2 # convert to 360 degrees space
        bad_trials1=dat['ft_mem_long'][0,0]['bad_trials']
        ntrls1=theta1.shape[0]
        time=np.squeeze(dat['ft_mem_long'][0,0]['time'])

        eeg_dat2=dat['ft_mem_short'][0,0]['trial']
        theta2=dat['ft_mem_short'][0,0]['theta_rad']*2 # convert to 360 degrees space
        bad_trials2=dat['ft_mem_short'][0,0]['bad_trials']
        ntrls2=theta2.shape[0]
        
        # combine trial types
        incl1=np.setdiff1d(np.array(range(1,ntrls1+1)),bad_trials1)-1
        incl2=np.setdiff1d(np.array(range(1,ntrls2+1)),bad_trials2)-1
        incl=np.concatenate((incl1,incl2+ntrls1),axis=0)
        theta=np.concatenate((theta1,theta2),axis=0)
        theta=theta[incl]
        eeg_dat=np.concatenate((eeg_dat1,eeg_dat2),axis=0)
        
        # change orientation labels so that 0 degrees is horizontal
        theta=theta+np.pi
        # flip direction so that orientations go counterclockwise
        theta_temp=np.copy(theta)
        theta[theta_temp>np.pi]=-theta[theta_temp>np.pi]+2*np.pi
        theta[theta_temp<np.pi]=-theta[theta_temp<np.pi]+2*np.pi

        # combine channel and temporal dimensions
        dat_4d=dat_prep_4d_section(eeg_dat[incl,:,:],time_dat=time,toi=toi,span=span,hz=hz,relative_baseline=True,in_ms=True)

        dec_acc,_,distances_ordered,_,angspace_full=dist_theta_kfold(dat_4d,theta,n_reps=n_reps,n_folds=n_folds,ang_steps=ang_steps)

        orient_4d_dec.append(dec_acc)
        orient_4d_dists.append(distances_ordered)
        thetas.append(theta)

    with open(results_dir+'/Orient_dec_4d_Wolff15.pickle','wb') as f:
        pickle.dump([orient_4d_dec,orient_4d_dists,thetas,angspace_full],f) 

#%% Decode orientation from Wolff et al. 2017    
if do_decoding_W17:
    hz=500
    span=20
    orient_left_4d_dec=[]; orient_left_4d_dists=[];  thetas_left=[]
    orient_right_4d_dec=[]; orient_right_4d_dists=[];  thetas_right=[]
    for isub in range(30):
        print(isub+1)
        
        dat=sio.loadmat(dat_dir+'/dat_2017_exp1_mem_items_' +str(isub+1)+'.mat')    
        ft_dat=dat['ft_mem']
        eeg_dat=dat['ft_mem'][0,0]['trial']
        theta_left=dat['ft_mem'][0,0]['theta_rad_left']*2
        theta_right=dat['ft_mem'][0,0]['theta_rad_right']*2 
        bad_trials=dat['ft_mem'][0,0]['bad_trials']
        ntrls=eeg_dat.shape[0]
        time=np.squeeze(dat['ft_mem'][0,0]['time'])
        incl=np.setdiff1d(np.array(range(1,ntrls+1)),bad_trials)-1
        
        theta_left= theta_left[incl]
        theta_right= theta_right[incl]

        # change orientation labels so that 0 degrees is horizontal
        theta_left=theta_left+np.pi
        theta_right=theta_right+np.pi
        # flip direction so that orientations go counterclockwise
        theta_temp=np.copy(theta_left)
        theta_left[theta_temp>np.pi]=-theta_left[theta_temp>np.pi]+2*np.pi
        theta_left[theta_temp<np.pi]=-theta_left[theta_temp<np.pi]+2*np.pi
        theta_temp=np.copy(theta_right)
        theta_right[theta_temp>np.pi]=-theta_right[theta_temp>np.pi]+2*np.pi
        theta_right[theta_temp<np.pi]=-theta_right[theta_temp<np.pi]+2*np.pi

        # combine channel and temporal dimensions
        dat_4d=dat_prep_4d_section(eeg_dat[incl,:,:],time_dat=time,toi=toi,span=span,hz=hz,relative_baseline=True,in_ms=True)

        # decode left and right orientations separately
        dec_acc,_,distances_ordered,_,angspace_full=dist_theta_kfold(dat_4d,theta_left,n_reps=n_reps,n_folds=n_folds,ang_steps=ang_steps)

        orient_left_4d_dec.append(dec_acc)
        orient_left_4d_dists.append(distances_ordered)
        thetas_left.append(theta_left)

        dec_acc,_,distances_ordered,_,angspace_full=dist_theta_kfold(dat_4d,theta_right,n_reps=n_reps,n_folds=n_folds,ang_steps=ang_steps)

        orient_right_4d_dec.append(dec_acc)
        orient_right_4d_dists.append(distances_ordered)
        thetas_right.append(theta_right)

    with open(results_dir+'/Orient_dec_4d_Wolff17.pickle','wb') as f:
        pickle.dump([orient_left_4d_dec,orient_left_4d_dists,thetas_left,orient_right_4d_dec,orient_right_4d_dists,thetas_right,angspace_full],f) 

#%% Decode orientation from Wolff et al. 2020
if do_decoding_W20:
    hz=500
    span=20
    orient_left_4d_dec=[]; orient_left_4d_dists=[];  thetas_left=[]
    orient_right_4d_dec=[]; orient_right_4d_dists=[];  thetas_right=[]
    for isub in range(26):
        print(isub+1)
        
        dat=sio.loadmat(dat_dir+'/dat_2020_mem_items_' +str(isub+1)+'.mat')    
        ft_dat=dat['ft_mem']
        eeg_dat=dat['ft_mem'][0,0]['trial']
        theta_left=dat['ft_mem'][0,0]['theta_rad_left']*2
        theta_right=dat['ft_mem'][0,0]['theta_rad_right']*2 
        bad_trials=dat['ft_mem'][0,0]['bad_trials']
        ntrls=eeg_dat.shape[0]
        time=np.squeeze(dat['ft_mem'][0,0]['time'])
        incl=np.setdiff1d(np.array(range(1,ntrls+1)),bad_trials)-1
        
        theta_left= theta_left[incl]
        theta_right= theta_right[incl]

        # change orientation labels so that 0 degrees is horizontal
        theta_left=theta_left+np.pi
        theta_right=theta_right+np.pi
        # flip direction so that orientations go counterclockwise
        theta_temp=np.copy(theta_left)
        theta_left[theta_temp>np.pi]=-theta_left[theta_temp>np.pi]+2*np.pi
        theta_left[theta_temp<np.pi]=-theta_left[theta_temp<np.pi]+2*np.pi
        theta_temp=np.copy(theta_right)
        theta_right[theta_temp>np.pi]=-theta_right[theta_temp>np.pi]+2*np.pi
        theta_right[theta_temp<np.pi]=-theta_right[theta_temp<np.pi]+2*np.pi

        # combine channel and temporal dimensions
        dat_4d=dat_prep_4d_section(eeg_dat[incl,:,:],time_dat=time,toi=toi,span=span,hz=hz,relative_baseline=True,in_ms=True)

        # decode left and right orientations separately
        dec_acc,_,distances_ordered,_,angspace_full=dist_theta_kfold(dat_4d,theta_left,n_reps=n_reps,n_folds=n_folds,ang_steps=ang_steps)

        orient_left_4d_dec.append(dec_acc)
        orient_left_4d_dists.append(distances_ordered)
        thetas_left.append(theta_left)

        dec_acc,_,distances_ordered,_,angspace_full=dist_theta_kfold(dat_4d,theta_right,n_reps=n_reps,n_folds=n_folds,ang_steps=ang_steps)

        orient_right_4d_dec.append(dec_acc)
        orient_right_4d_dists.append(distances_ordered)
        thetas_right.append(theta_right)

    with open(results_dir+'/Orient_dec_4d_Wolff20.pickle','wb') as f:
        pickle.dump([orient_left_4d_dec,orient_left_4d_dists,thetas_left,orient_right_4d_dec,orient_right_4d_dists,thetas_right,angspace_full],f) 

#%% load in results from all experimentsand plot them

# make orientation bins, the bin witdh is 22.5 and we have 16 time 8 bins in total (so there is overlap between bins!)
ang_bins_temp=np.arange(0,2*np.pi,np.pi/8)
bin_width=np.diff(ang_bins_temp)[0]
ang_bins=np.zeros((8,len(ang_bins_temp)))
for i in range(8):
    ang_bins[i,:]=ang_bins_temp+i*bin_width/8

ang_bins_deg=np.round(np.rad2deg(np.reshape(ang_bins,(ang_bins.shape[0]*ang_bins.shape[1]),order='F')),15)

for iexp in range(4):
    if iexp==0:
        with open(results_dir+'/Orient_dec_4d_Harrison23.pickle','rb') as f:
            orient_4d_dec,orient_4d_dists,thetas,angspace_full=pickle.load(f)  
    elif iexp==1:
        with open(results_dir+'/Orient_dec_4d_Wolff15.pickle','rb') as f:
            orient_4d_dec,orient_4d_dists,thetas,angspace_full=pickle.load(f)
    elif iexp==2:
        with open(results_dir+'/Orient_dec_4d_Wolff17.pickle','rb') as f:
            orient_left_4d_dec,orient_left_4d_dists,thetas_left,orient_right_4d_dec,orient_right_4d_dists,thetas_right,angspace_full=pickle.load(f)
    elif iexp==3:
        with open(results_dir+'/Orient_dec_4d_Wolff20.pickle','rb') as f:
            orient_left_4d_dec,orient_left_4d_dists,thetas_left,orient_right_4d_dec,orient_right_4d_dists,thetas_right,angspace_full=pickle.load(f)
    
    # combine left and right orientations for Wolff et al. 2017 and 2020
    if iexp>1:
        thetas=[]
        orient_4d_dec=[]
        orient_4d_dists=[]
        nsubs=len(orient_left_4d_dec)
        for sub in range(nsubs):
            orient_4d_dec.append(np.concatenate((orient_left_4d_dec[sub],orient_right_4d_dec[sub]),axis=0))
            orient_4d_dists.append(np.concatenate((orient_left_4d_dists[sub],orient_right_4d_dists[sub]),axis=1))
            thetas.append(np.concatenate((thetas_left[sub],thetas_right[sub]),axis=0))

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

    #%
    vert_bins=[67.5,112.5]
    horz1_bins=[0,22.5]
    horz2_bins=[157.5,180]

    horz1_dir_bins=[11.25,45-11.25]
    horz2_dir_bins=[135+11.25,180-11.25]
    vert1_dir_bins=[45+11.25,90-11.25]
    vert2_dir_bins=[90+11.25,135-11.25]

    angspace_ang=np.rad2deg(angspace_full)/2+90

    angspace_vert_ind=np.where((angspace_ang>vert_bins[0]+.1)&(angspace_ang<vert_bins[1]-.1))[0]
    angspace_vert=angspace_ang[angspace_vert_ind]

    angspace_horz1_ind=np.where((angspace_ang<horz1_bins[1]-.1))[0]
    angspace_horz1=angspace_ang[angspace_horz1_ind]

    angspace_horz2_ind=np.where((angspace_ang>horz2_bins[0]+.1))[0]
    angspace_horz2=angspace_ang[angspace_horz2_ind]

    # select vertical orientations for dec_bins_mc
    dec_bins_mc_vert=dec_bins_mc[:,angspace_vert_ind]

    # select horizontal orientations for dec_bins_mc
    dec_bins_mc_horz=np.concatenate((dec_bins_mc[:,angspace_horz1_ind],dec_bins_mc[:,angspace_horz2_ind]),axis=1)

    # select vertical orientations for prec_bins_mc
    prec_bins_mc_vert=prec_bins_mc[:,angspace_vert_ind]

    # select horizontal orientations for prec_bins_mc
    prec_bins_mc_horz=np.concatenate((prec_bins_mc[:,angspace_horz1_ind],prec_bins_mc[:,angspace_horz2_ind]),axis=1)

    # select bins for dir_bins
    angspace_horz_attr_neg_ind=np.where((angspace_ang>horz1_dir_bins[0]+.1)&(angspace_ang<horz1_dir_bins[1]-.1))[0]
    angspace_horz_attr_neg=angspace_ang[angspace_horz_attr_neg_ind]

    angspace_horz_attr_pos_ind=np.where((angspace_ang>horz2_dir_bins[0]+.1)&(angspace_ang<horz2_dir_bins[1]-.1))[0]
    angspace_horz_attr_pos=angspace_ang[angspace_horz_attr_pos_ind]

    angspace_vert_attr_pos_ind=np.where((angspace_ang>vert1_dir_bins[0]+.1)&(angspace_ang<vert1_dir_bins[1]-.1))[0]
    angspace_vert_attr_pos=angspace_ang[angspace_vert_attr_pos_ind]

    angspace_vert_attr_neg_ind=np.where((angspace_ang>vert2_dir_bins[0]+.1)&(angspace_ang<vert2_dir_bins[1]-.1))[0]
    angspace_vert_attr_neg=angspace_ang[angspace_vert_attr_neg_ind]

    # select horizontal orientations for dir_bins
    dir_bins_horz_neg=dir_bins[:,angspace_horz_attr_neg_ind]
    dir_bins_horz_pos=dir_bins[:,angspace_horz_attr_pos_ind]

    # select vertical orientations for dir_bins
    dir_bins_vert_neg=dir_bins[:,angspace_vert_attr_neg_ind]
    dir_bins_vert_pos=dir_bins[:,angspace_vert_attr_pos_ind]


    # compute differences between horizontal and vertical orientations
    dec_horz_vert_diff=np.mean(dec_bins_mc_horz,axis=1)-np.mean(dec_bins_mc_vert,axis=1)
    prec_horz_vert_diff=np.mean(prec_bins_mc_horz,axis=1)-np.mean(prec_bins_mc_vert,axis=1)

    dir_horz_vert_diff=(-np.mean(dir_bins_horz_neg,axis=1)+np.mean(dir_bins_horz_pos,axis=1))/2-(-np.mean(dir_bins_vert_neg,axis=1)+np.mean(dir_bins_vert_pos,axis=1))/2
    #%
    _,pd_dec,_=permutation_t_test(np.expand_dims(dec_horz_vert_diff,axis=-1), n_permutations=100000,verbose=False)
    _,pd_prec,_=permutation_t_test(np.expand_dims(prec_horz_vert_diff,axis=-1), n_permutations=100000,verbose=False)
    _,pd_dir,_=permutation_t_test(np.expand_dims(dir_horz_vert_diff,axis=-1), n_permutations=100000,verbose=False  )

    print('decoding accuracy: p = '+str(pd_dec[0]))
    print('precision: p = '+str(pd_prec[0]))
    print('direction: p = '+str(pd_dir[0]))

    # convert everything to pandas dataframe for plotting with seaborn
    dec_horz_vert_diff_df=pd.DataFrame(dec_horz_vert_diff)
    dec_horz_vert_diff_df.columns=['decoding accuracy']

    prec_horz_vert_diff_df=pd.DataFrame(prec_horz_vert_diff)
    prec_horz_vert_diff_df.columns=['precision']

    dir_horz_vert_diff_df=pd.DataFrame(dir_horz_vert_diff)
    dir_horz_vert_diff_df.columns=['direction']

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
    vert_color='purple'
    horz_color='green'

    dec_acc_ylim=[-0.0065,0.0065]
    dec_acc_diff_ylim=[-0.0065,0.0065]
    prec_ylim=[-0.175,0.175]
    prec_diff_ylim=[-0.175,0.175]
    dir_ylim=[-20,20]
    dir_diff_ylim=[-20,20]

    n_boot=10000
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(4.9, 7),
                        gridspec_kw={
                            'width_ratios': [1.5, .25],
                            'height_ratios': [1,1,1],
                        'wspace': .6,
                        'hspace': .1})
    if iexp==0:
        plt.suptitle('Central orientation, Harrison et al. 2023')
    elif iexp==1:
        plt.suptitle('Central orientation, Wolff et al. 2015')
    elif iexp==2:
        plt.suptitle('Lateral orientations, Wolff et al. 2017')
    elif iexp==3:
        plt.suptitle('Lateral orientations, Wolff et al. 2020')
    plt.axes(ax[0,0])
    sns.lineplot(x='orientation',y='decoding accuracy',data=dec_bins_mc_smooth_df,errorbar=('ci',95),n_boot=n_boot)
    # plot vertical line at 0
    plt.axvline(90,color='k',linestyle='--',linewidth=1)
    # plot horizontal line at 0
    plt.axhline(0,color='k',linestyle='--',linewidth=1)
    plt.yticks(np.arange(-.01,.011,.005))
    # make shaded area for vertical orientations
    plt.fill_between(angspace_vert,dec_acc_ylim[0],np.mean(dec_bins_mc_smooth[:,angspace_vert_ind],axis=0), alpha=0.2, color=vert_color,zorder=0)

    # make shaded area for horizontal orientations1
    plt.fill_between(angspace_horz1,dec_acc_ylim[0],np.mean(dec_bins_mc_smooth[:,angspace_horz1_ind],axis=0), alpha=0.2, color=horz_color,zorder=0)

    # make shaded area for horizontal orientations2
    plt.fill_between(angspace_horz2,dec_acc_ylim[0],np.mean(dec_bins_mc_smooth[:,angspace_horz2_ind],axis=0), alpha=0.2, color=horz_color,zorder=0)

    plt.xlim([0,180])
    plt.ylim(dec_acc_ylim)
    plt.xticks(np.arange(0,181,22.5),[])
    plt.xlabel('')
    plt.ylabel('Relative decoding accuracy')

    plt.axes(ax[0,1])
    sns.boxplot(data=dec_horz_vert_diff_df, width=.4,showfliers = False,zorder=1,dodge=False)
    sns.stripplot(data=dec_horz_vert_diff_df,size=4,zorder=2,dodge=False,palette='dark',jitter=.15,edgecolor='white',linewidth=.25)
    # ax.set_aspect(aspect=5)
    sns.pointplot(color='black',data=dec_horz_vert_diff_df,errorbar=('ci', 95),n_boot=n_boot,capsize=.15, scale = .9,zorder=100)
    plt.axhline(0,color='k',linestyle='--',linewidth=1)
    plt.yticks(np.arange(-.01,.011,.005))
    plt.ylim(dec_acc_diff_ylim)
    plt.xticks([])
    if pd_dec<.05:
        plt.annotate('*',xy=(0,np.max(dec_horz_vert_diff)+0.0003),xycoords='data',fontsize=20,verticalalignment='center',horizontalalignment='center')
    plt.ylabel('horizontal minus vertical')
    # plt.tight_layout()

    plt.axes(ax[1,0])
    sns.lineplot(x='orientation',y='precision',data=prec_bins_mc_smooth_df,errorbar=('ci',95),n_boot=n_boot)
    # plot vertical line at 0
    plt.axvline(90,color='k',linestyle='--',linewidth=1)
    # plot horizontal line at 0
    plt.axhline(0,color='k',linestyle='--',linewidth=1)

    # make shaded area for vertical orientations
    plt.fill_between(angspace_vert,prec_ylim[0],np.mean(prec_bins_mc_smooth[:,angspace_vert_ind],axis=0), alpha=0.2, color=vert_color,zorder=0)

    # make shaded area for horizontal orientations1
    plt.fill_between(angspace_horz1,prec_ylim[0],np.mean(prec_bins_mc_smooth[:,angspace_horz1_ind],axis=0), alpha=0.2, color=horz_color,zorder=0)

    # make shaded area for horizontal orientations2
    plt.fill_between(angspace_horz2,prec_ylim[0],np.mean(prec_bins_mc_smooth[:,angspace_horz2_ind],axis=0), alpha=0.2, color=horz_color,zorder=0)

    plt.xlim([0,180])
    plt.ylim(prec_ylim)
    plt.xticks(np.arange(0,181,22.5),[])
    plt.xlabel('')
    plt.ylabel('Relative precision')

    plt.axes(ax[1,1])
    sns.boxplot(data=prec_horz_vert_diff_df, width=.4,showfliers = False,zorder=1,dodge=False)
    sns.stripplot(data=prec_horz_vert_diff_df,size=4,zorder=2,dodge=False,palette='dark',jitter=.15,edgecolor='white',linewidth=.25)
    # ax.set_aspect(aspect=5)
    sns.pointplot(color='black',data=prec_horz_vert_diff_df,errorbar=('ci', 95),n_boot=n_boot,capsize=.15, scale = .9,zorder=100)
    plt.axhline(0,color='k',linestyle='--',linewidth=1)
    plt.ylim(prec_diff_ylim)
    plt.xticks([])
    if pd_prec<.05:
        plt.annotate('*',xy=(0,np.max(prec_horz_vert_diff)+0.02),xycoords='data',fontsize=20,verticalalignment='center',horizontalalignment='center')
    plt.ylabel('horizontal minus vertical')

    plt.axes(ax[2,0])
    sns.lineplot(x='orientation',y='direction',data=dir_bins_smooth_df,errorbar=('ci',95),n_boot=n_boot)
    # plot vertical line at 0
    plt.axvline(90,color='k',linestyle='--',linewidth=1)
    # plot horizontal line at 0
    plt.axhline(0,color='k',linestyle='--',linewidth=1)

    # make shaded area for vertical orientations1
    plt.fill_between(angspace_vert_attr_pos,dir_ylim[0],np.mean(dir_bins_smooth[:,angspace_vert_attr_pos_ind],axis=0), alpha=0.2, color=vert_color,zorder=0)

    # make shaded area for vertical orientations2
    plt.fill_between(angspace_vert_attr_neg,dir_ylim[1],np.mean(dir_bins_smooth[:,angspace_vert_attr_neg_ind],axis=0), alpha=0.2, color=vert_color,zorder=0)

    # make shaded area for horizontal orientations1
    plt.fill_between(angspace_horz_attr_neg,dir_ylim[1],np.mean(dir_bins_smooth[:,angspace_horz_attr_neg_ind],axis=0), alpha=0.2, color=horz_color,zorder=0)

    # make shaded area for horizontal orientations2
    plt.fill_between(angspace_horz_attr_pos,dir_ylim[0],np.mean(dir_bins_smooth[:,angspace_horz_attr_pos_ind],axis=0), alpha=0.2, color=horz_color,zorder=0)

    plt.xlim([0,180])
    plt.ylim(dir_ylim)
    plt.xticks(np.arange(0,181,22.5),rotation=-45)
    plt.xlabel('Orientation (deg)')
    plt.ylabel('Bias (deg)')

    plt.axes(ax[2,1])
    sns.boxplot(data=dir_horz_vert_diff_df, width=.4,showfliers = False,zorder=1,dodge=False)
    sns.stripplot(data=dir_horz_vert_diff_df,size=4,zorder=2,dodge=False,palette='dark',jitter=.15,edgecolor='white',linewidth=.25)
    # ax.set_aspect(aspect=5)
    sns.pointplot(color='black',data=dir_horz_vert_diff_df,errorbar=('ci', 95),n_boot=n_boot,capsize=.15, scale = .9,zorder=100)
    plt.axhline(0,color='k',linestyle='--',linewidth=1)
    plt.ylim(dir_diff_ylim)
    plt.xticks([])
    if pd_dir<.05:
        plt.annotate('*',xy=(0,np.min(dir_horz_vert_diff)-3.5),xycoords='data',fontsize=20,verticalalignment='center',horizontalalignment='center')
    plt.ylabel('horizontal minus vertical')

    if iexp==0:
        fig_name='/Orient_dec_4d_Harrison23'
    elif iexp==1:
        fig_name='/Orient_dec_4d_Wolff15'
    elif iexp==2:
        fig_name='/Orient_dec_4d_Wolff17'
    elif iexp==3:
        fig_name='/Orient_dec_4d_Wolff20'

    # # save figure as .svg
    if save_fig:
        plt.savefig(fig_dir_out+fig_name+ '.svg',dpi=300,bbox_inches='tight')

    # plt.show()


        