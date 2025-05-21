#%% 
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
import matplotlib as mpl

dat_dir_out=''

# variables that are consistent across experiments by Foster et al. 2015 and 2017
post_chans = [ 'PO3','PO4','P3','P4','O1','O2','POz','Pz'] # channels to be included
toi=np.asarray([.052, .452]) # time window of interest
hz=250 # sampling rate of data
span=20 # length of each time bin (in ms) when creating 4d data
ang_steps=8 # number of times to shif the orientations bins when decoding
n_reps=20 # number of repetitions for the decoder (with random folds each time)
n_folds=8 # number of folds 

relative_change=True # if True, plot relative change in decoding accuracy from average, otherwise we simply mean center

do_decoding_15_1=False
do_decoding_15_2=False
do_decoding_15_3=False
do_decoding_17_1=False
do_decoding_21_1=False
#%% Decode location from Foster et al. 2015, Exp. 1
if do_decoding_15_1:
    
    time=np.arange(-1,2.504,1/hz) # time vector

    # data is publicly available at https://osf.io/bwzfj/ from the original authors
    dat_dir='' # location of data 
    
    eeg_files=[]; behav_files=[]
    for file in os.listdir(dat_dir):
        if file.endswith('_EEG.mat'):
            eeg_files.append(file)
        if file.endswith('_MixModel_wBias.mat'):
            behav_files.append(file)
    eeg_files.sort(); behav_files.sort()
    files = zip(eeg_files,behav_files)

    loc_4d_dec=[]; thetas=[]
    for isub,file in enumerate(files):
        print(isub+1)
        
        eeg_file=file[0]
        behav_file=file[1]

        dat=mat73.loadmat(dat_dir+eeg_file)['eeg'] # load EEG data
        eeg_dat=dat['data']
        labels=dat['chanLabels']
        bad_trials=dat['arf']['artIndCleaned']
        incl_chans_post = np.in1d(labels, post_chans) # indices of to-be-included channels
        eeg_dat=eeg_dat[~np.squeeze(bad_trials),:,:][:,incl_chans_post,:]
        dat_4d=dat_prep_4d_section(eeg_dat,time_dat=time,toi=toi,span=span,hz=hz)

        behav=sio.loadmat(dat_dir+behav_file)['beh'] # load behavioral data
        theta=np.squeeze(np.deg2rad(behav['trial'][0,0]['pos'][0,0])) # location in radians
        theta=theta[~np.squeeze(bad_trials)] # remove bad trials

        # locations are labelled such that 0 is on the right (that's what we want), but they move clockwise (we don't want that)
        # so we flip the direction
        theta_temp=np.copy(theta)
        theta[theta_temp>np.pi]=-theta[theta_temp>np.pi]+2*np.pi
        theta[theta_temp<np.pi]=-theta[theta_temp<np.pi]+2*np.pi

        dec_cos,distances,distances_ordered,angspaces,angspace_full=dist_theta_kfold(dat_4d,theta,n_reps=n_reps,n_folds=n_folds,ang_steps=ang_steps)

        loc_4d_dec.append(dec_cos)
        thetas.append(theta)

    with open(dat_dir_out+'/Loc_dec_4d_Foster15_1.pickle','wb') as f:
        pickle.dump([loc_4d_dec,thetas,angspaces,angspace_full],f)

#%% Decode location from Foster et al. 2015, Exp. 2
if do_decoding_15_2:
    
    time=np.arange(-1,2.504,1/hz) # time vector

    # data is publicly available at https://osf.io/bwzfj/ from the original authors
    dat_dir='' # location of data 
    
    eeg_files=[]; behav_files=[]
    for file in os.listdir(dat_dir):
        if file.endswith('_EEG.mat'):
            eeg_files.append(file)
        if file.endswith('_MixModel_wBias.mat'):
            behav_files.append(file)
    eeg_files.sort(); behav_files.sort()
    files = zip(eeg_files,behav_files)

    loc_4d_dec=[]; thetas=[]
    for isub,file in enumerate(files):
        print(isub+1)
        
        eeg_file=file[0]
        behav_file=file[1]

        dat=mat73.loadmat(dat_dir+eeg_file)['eeg'] # load EEG data
        eeg_dat=dat['data']
        labels=dat['chanLabels']
        bad_trials=dat['arf']['artIndCleaned']
        incl_chans_post = np.in1d(labels, post_chans) # indices of to-be-included channels
        eeg_dat=eeg_dat[~np.squeeze(bad_trials),:,:][:,incl_chans_post,:]
        dat_4d=dat_prep_4d_section(eeg_dat,time_dat=time,toi=toi,span=span,hz=hz)

        behav=sio.loadmat(dat_dir+behav_file)['beh'] # load behavioral data
        theta=np.squeeze(np.deg2rad(behav['trial'][0,0]['pos'][0,0])) # location in radians
        theta=theta[~np.squeeze(bad_trials)] # remove bad trials

        # locations are labelled such that 0 is on the right (that's what we want), but they move clockwise (we don't want that)
        # so we flip the direction
        theta_temp=np.copy(theta)
        theta[theta_temp>np.pi]=-theta[theta_temp>np.pi]+2*np.pi
        theta[theta_temp<np.pi]=-theta[theta_temp<np.pi]+2*np.pi

        dec_cos,distances,distances_ordered,angspaces,angspace_full=dist_theta_kfold(dat_4d,theta,n_reps=n_reps,n_folds=n_folds,ang_steps=ang_steps)

        loc_4d_dec.append(dec_cos)
        thetas.append(theta)

    with open(dat_dir_out+'/Loc_dec_4d_Foster15_2.pickle','wb') as f:
        pickle.dump([loc_4d_dec,thetas,angspaces,angspace_full],f)

#%% Decode location from Foster et al. 2015, Exp. 3
if do_decoding_15_3:
    
    time=np.arange(-1,2.504,1/hz) # time vector

    # data is publicly available at https://osf.io/bwzfj/ from the original authors
    dat_dir='' # location of data 
    
    eeg_files=[]; behav_files=[]
    for file in os.listdir(dat_dir):
        if file.endswith('_EEG.mat'):
            eeg_files.append(file)
        if file.endswith('_ChangeDetect_Behavior_Fixed.mat'):
            behav_files.append(file)
    eeg_files.sort(); behav_files.sort()
    files = zip(eeg_files,behav_files)

    loc_4d_dec=[]; thetas=[]
    for isub,file in enumerate(files):
        print(isub+1)
        
        eeg_file=file[0]
        behav_file=file[1]

        # EEG data file of 17_EEG.mat is in a different format than the others
        if eeg_file[:2]=='17':
            dat=sio.loadmat(dat_dir+eeg_file)['eeg']
            eeg_dat=dat['data'][0,0]
            labels=dat['chanLabels'][0,0]
            bad_trials=dat['arf'][0,0]['artIndCleaned'][0,0]
            bad_trials=np.array(bad_trials,dtype=bool)
        else:
            dat=mat73.loadmat(dat_dir+eeg_file)['eeg']
            eeg_dat=dat['data']
            labels=dat['chanLabels']
            bad_trials=dat['arf']['artIndCleaned']
        incl_chans_post = np.in1d(labels, post_chans) # indices of to-be-included channels
        eeg_dat=eeg_dat[~np.squeeze(bad_trials),:,:][:,incl_chans_post,:]
        dat_4d=dat_prep_4d_section(eeg_dat,time_dat=time,toi=toi,span=span,hz=hz)

        behav=sio.loadmat(dat_dir+behav_file)['beh'] # load behavioral data
        theta=np.squeeze(np.deg2rad(behav['trial'][0,0]['sampArc'][0,0])) # location in radians
        theta=theta[~np.squeeze(bad_trials)] # remove bad trials

        # the original notation in this dataset is 0 vertical bottom (we don't want that), then moving counter-clockwise (we want that)
        theta=theta-np.pi/2
        theta[theta<0]=theta[theta<0]+2*np.pi

        dec_cos,distances,distances_ordered,angspaces,angspace_full=dist_theta_kfold(dat_4d,theta,n_reps=n_reps,n_folds=n_folds,ang_steps=ang_steps)

        loc_4d_dec.append(dec_cos)
        thetas.append(theta)

    with open(dat_dir_out+'/Loc_dec_4d_Foster15_3.pickle','wb') as f:
        pickle.dump([loc_4d_dec,thetas,angspaces,angspace_full],f)

#%% Decode location from Foster et al. 2017, Exp. 1 (the only experiment with only a single item on the screen)

if do_decoding_17_1:
    
    time=np.arange(-.8,1.804,1/250) # time vector

    # data is publicly available at https://osf.io/vw4uc/ from the original authors
    dat_dir='' # location of data 
    
    eeg_files=[]; behav_files=[]
    for file in os.listdir(dat_dir):
        if file.endswith('_EEG.mat'):
            # don't include subject 1 and 8, they are excluded in the original paper as well
            if file!='1_EEG.mat' and file!='8_EEG.mat':
                eeg_files.append(file)
        if file.endswith('_Behavior.mat'):
            if file!='1_Behavior.mat' and file!='8_Behavior.mat':
                behav_files.append(file)
    eeg_files.sort(); behav_files.sort()
    files = zip(eeg_files,behav_files)

    loc_4d_dec=[]; thetas=[]
    for isub,file in enumerate(files):
        print(isub+1)
        
        eeg_file=file[0]
        behav_file=file[1]

        dat=mat73.loadmat(dat_dir+eeg_file)['eeg'] # load EEG data
        eeg_dat=dat['data']
        labels=dat['arf']['chanLabels']
        bad_trials=dat['arf']['artifactIndCleaned']
        incl_chans_post = np.in1d(labels, post_chans) # indices of to-be-included channels
        eeg_dat=eeg_dat[~np.squeeze(bad_trials),:,:][:,incl_chans_post,:]
        dat_4d=dat_prep_4d_section(eeg_dat,time_dat=time,toi=toi,span=span,hz=hz)

        behav=sio.loadmat(dat_dir+behav_file)['beh'] # load behavioral data
        theta=np.squeeze(np.deg2rad(behav[0,0]['pos'])) # location in radians
        theta=theta[~np.squeeze(bad_trials)] # remove bad trials

        # locations are labelled such that 0 is on the right (that's what we want), but they move clockwise (we don't want that)
        # so we flip the direction
        theta_temp=np.copy(theta)
        theta[theta_temp>np.pi]=-theta[theta_temp>np.pi]+2*np.pi
        theta[theta_temp<np.pi]=-theta[theta_temp<np.pi]+2*np.pi

        dec_cos,distances,distances_ordered,angspaces,angspace_full=dist_theta_kfold(dat_4d,theta,n_reps=n_reps,n_folds=n_folds,ang_steps=ang_steps)

        loc_4d_dec.append(dec_cos)
        thetas.append(theta)

    with open(dat_dir_out+'/Loc_dec_4d_Foster17_1.pickle','wb') as f:
        pickle.dump([loc_4d_dec,thetas,angspaces,angspace_full],f)

#%% Decoding location from Bae experiment 1 2021, note that here we have 16 discrete locations
dat_dir=''
os.listdir(dat_dir)
prefixed = [filename for filename in os.listdir(dat_dir) if filename.startswith('SWM_NI_Exp1_')]
nsubs=len(prefixed)

# the labels as they are ordered in the data
eeg_labels=['FP1','Fz','F3','F7','Cz','C3','Pz','P3','P5','P7','P9','PO7','PO3','O1','POz','Oz','FP2','F4','F8','C4','P4','P6','P8','P10','PO4','PO8','O2','AF3','AF7','F1','F5',
'FCz','FC1','FC3','FC5','C1','C5','T7','TP7','CP1','CP3','CP5','P1','AF4','AF8','F2','F6','FC2','FC4','FC6','C2','C6','T8','CP6','TP8','CPz','CP2','CP4','P2','HEOG R-L','VEOG Lower-Upper','Photosensor']

post_chans = [ 'P7','P5','P3','P1','Pz','P2','P4','P6','P8','PO7','PO3','POz','PO4','PO8','O2','O1','Oz']

time=np.arange(-.5,1.5,1/250)

loc_4d_dec=[]; thetas=[]
if do_decoding_21_1:
    for isub, subfile in enumerate(prefixed):
        print(isub+1)
        
        dat_temp=sio.loadmat(dat_dir+subfile) 
        dat=dat_temp['data']
        eeg_dat=dat[0,0]['eeg']
        theta_ang=dat[0,0]['locationAngle']
        theta=np.squeeze(np.deg2rad(theta_ang))   
        incl_chans_post = np.in1d(eeg_labels, post_chans) # indices of to-be-included channels
        eeg_dat=eeg_dat[:,incl_chans_post,:]
        dat_4d=dat_prep_4d_section(eeg_dat,time_dat=time,toi=toi,span=span,hz=hz,relative_baseline=True,in_ms=True)

        # locations are labelled such that 0 is on the right (that's what we want), but they move clockwise (we don't want that)
        # so we flip the direction
        theta_temp=np.copy(theta)
        theta[theta_temp>np.pi]=-theta[theta_temp>np.pi]+2*np.pi
        theta[theta_temp<np.pi]=-theta[theta_temp<np.pi]+2*np.pi

        dec_acc,_,_,_,_=dist_theta_kfold(dat_4d,theta,n_reps=n_reps,n_folds=n_folds,ang_steps=1)
       
        loc_4d_dec.append(dec_acc)
        thetas.append(theta)

    with open(dat_dir_out+'/Loc_dec_4d_Bae21_1.pickle','wb') as f:
        pickle.dump([loc_4d_dec,thetas],f)  

#%% load in results from all experiments by Foster, and plot them

# make location bins, the bin witdh is 22.5 and we have 16 time 8 bins in total (so there is overlap between bins!)
ang_bins_temp=np.arange(0,2*np.pi,np.pi/8)
bin_width=np.diff(ang_bins_temp)[0]
ang_bins=np.zeros((8,len(ang_bins_temp)))
for i in range(8):
    ang_bins[i,:]=ang_bins_temp+i*bin_width/8

ang_bins_full_deg=np.round(np.rad2deg(np.reshape(ang_bins,(ang_bins.shape[0]*ang_bins.shape[1]),order='F')),15)

for iexp in range(4):
    if iexp==0:
        with open(dat_dir_out+'/Loc_dec_4d_Foster15_1.pickle','rb') as f:
            loc_4d_dec,thetas=pickle.load(f)  
    elif iexp==1:
        with open(dat_dir_out+'/Loc_dec_4d_Foster15_2.pickle','rb') as f:
            loc_4d_dec,thetas=pickle.load(f)
    elif iexp==2:
        with open(dat_dir_out+'/Loc_dec_4d_Foster15_3.pickle','rb') as f:
            loc_4d_dec,thetas=pickle.load(f)
    elif iexp==3:
        with open(dat_dir_out+'/Loc_dec_4d_Foster17_1.pickle','rb') as f:
            loc_4d_dec,thetas=pickle.load(f)
    
    nsubs=len(loc_4d_dec)
    dec_bins=np.zeros((nsubs,ang_bins.shape[0],ang_bins.shape[1]))

    for isub in range(nsubs):
        dec_temp=loc_4d_dec[isub]
        theta_temp=thetas[isub]

        for abin in range(ang_bins.shape[0]):
            ang_bin_temp=ang_bins[abin,:]

            # bin locations into current bin space
            temp=np.argmin(abs(circ_dist(ang_bin_temp,theta_temp,all_pairs=True)),axis=1)
            ang_bin=np.tile(ang_bin_temp,(len(theta_temp),1))        
            theta_bins=ang_bin[:,temp][0,:]

            for ibin,bin in enumerate(ang_bin_temp):
                dec_bins[isub,abin,ibin]=np.mean(dec_temp[theta_bins==bin])
    
    dec_bins=np.reshape(dec_bins,(dec_bins.shape[0],dec_bins.shape[1]*dec_bins.shape[2]),order='F')

    if relative_change:
        dec_bins_mc=(dec_bins-np.mean(dec_bins,axis=1,keepdims=True))/np.mean(dec_bins,axis=1,keepdims=True)
    else:
        dec_bins_mc=dec_bins-np.mean(dec_bins,axis=1,keepdims=True) # convert to relative decoding accuracy by mean centering


    # smooth across bins
    sigma=2/(180/len(ang_bins_full_deg))
    dec_bins_mc_smooth=np.concatenate((dec_bins_mc,dec_bins_mc,dec_bins_mc),axis=1)
    dec_bins_mc_smooth=gaussian_filter1d(dec_bins_mc_smooth,sigma=sigma,axis=1)
    dec_bins_mc_smooth=dec_bins_mc_smooth[:,dec_bins_mc.shape[1]:2*dec_bins_mc.shape[1]]

    if iexp==0:
        dec_bins_15_1_df=pd.DataFrame(dec_bins_mc_smooth)
    elif iexp==1:
        dec_bins_15_2_df=pd.DataFrame(dec_bins_mc_smooth)
    elif iexp==2:
        dec_bins_15_3_df=pd.DataFrame(dec_bins_mc_smooth)
    elif iexp==3:
        dec_bins_17_1_df=pd.DataFrame(dec_bins_mc_smooth)

experiments=['Foster 2015, Exp. 1','Foster 2015, Exp. 2','Foster 2015, Exp. 3','Foster 2017, Exp. 1']

# combine all experiments into a single dataframe
dec_bins_df=pd.concat([dec_bins_15_1_df,dec_bins_15_2_df,dec_bins_15_3_df,dec_bins_17_1_df],axis=0)
dec_bins_mc_smooth_Foster=dec_bins_df.to_numpy()
dec_bins_df.columns=ang_bins_full_deg
dec_bins_df['subject']=np.arange(1,dec_bins_df.shape[0]+1)
dec_bins_df['Experiment']=np.concatenate((np.tile(experiments[0],dec_bins_15_1_df.shape[0]),np.tile(experiments[1],dec_bins_15_2_df.shape[0]),np.tile(experiments[2],dec_bins_15_3_df.shape[0]),np.tile(experiments[3],dec_bins_17_1_df.shape[0])),axis=0)

# convert to long format
dec_bins_df=pd.melt(dec_bins_df,id_vars=['subject','Experiment'],var_name='orientation',value_name='decoding accuracy')
#%% define vertical and horizontal bins for plotting
bin_width_deg=45
horz1_bins1=[0,0+bin_width_deg/2]
horz2_bins1=[360-bin_width_deg/2,360]
horz_bins2=[180-bin_width_deg/2,180+bin_width_deg/2]
vert_bins1=[90-bin_width_deg/2,90+bin_width_deg/2]
vert_bins2=[270-bin_width_deg/2,270+bin_width_deg/2]

angspace_vert1_ind=np.where((ang_bins_full_deg>vert_bins1[0]+.1)&(ang_bins_full_deg<vert_bins1[1]-.1))[0]
angspace_vert1=ang_bins_full_deg[angspace_vert1_ind]
angspace_vert2_ind=np.where((ang_bins_full_deg>vert_bins2[0]+.1)&(ang_bins_full_deg<vert_bins2[1]-.1))[0]
angspace_vert2=ang_bins_full_deg[angspace_vert2_ind]
angspace_horz1_ind1=np.where((ang_bins_full_deg<horz1_bins1[1]-.1))[0]
angspace_horz1_1=ang_bins_full_deg[angspace_horz1_ind1]
angspace_horz1_ind2=np.where((ang_bins_full_deg>horz2_bins1[0]+.1))[0]
angspace_horz1_2=ang_bins_full_deg[angspace_horz1_ind2]
angspace_horz2_ind=np.where((ang_bins_full_deg>horz_bins2[0]+.1)&(ang_bins_full_deg<horz_bins2[1]-.1))[0]
angspace_horz2=ang_bins_full_deg[angspace_horz2_ind]
#%% since the data by Bae is not continuous, we plot that dat aseparately here and then combine the figures in Illustrator later
# plot all experiments by Foster et al. 2015 and 2017
vert_color='purple'
horz_color='green'

if relative_change:
    dec_acc_ylim=[-0.3,0.3]
else:
    dec_acc_ylim=[-0.012,0.012]

n_boot=1000
plt.figure(figsize=(4,3))
plt.suptitle('Location Decoding, Foster et al. 2015, 2017') 
# plt.axes(ax[0])


sns.lineplot(x='orientation',y='decoding accuracy',hue='Experiment',data=dec_bins_df,ci=None,linewidth=1,alpha=.5)
plt.axvline(90,color='k',linestyle='--',linewidth=1)
plt.axvline(180,color='k',linestyle='--',linewidth=1)
plt.axvline(270,color='k',linestyle='--',linewidth=1)
# plot horizontal line at 0
plt.axhline(0,color='k',linestyle='--',linewidth=1)
if relative_change:
    plt.yticks(np.arange(-.3,.301,.1))
else:
    plt.yticks(np.arange(-.01,.011,.005))

# place legend outside of the plot
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8, title='Experiment', title_fontsize=10)

plt.xlim([0,360])
plt.ylim(dec_acc_ylim)
plt.xticks(np.arange(0,361,45),rotation=-45)
plt.xlabel('Location (deg)')
plt.ylabel('Relative decoding accuracy')

plt. show()
#%% load in results from ALL experiments, combine them (with reduced binning), and plot them
# I know, this is qutie messy...

# make location bins
ang_bins_temp=np.arange(0,2*np.pi,np.pi/8)
bin_width=np.diff(ang_bins_temp)[0]

for iexp in range(5):
    if iexp==0:
        with open(dat_dir_out+'/Loc_dec_4d_Foster15_1.pickle','rb') as f:
            loc_4d_dec,thetas=pickle.load(f)  
    elif iexp==1:
        with open(dat_dir_out+'/Loc_dec_4d_Foster15_2.pickle','rb') as f:
            loc_4d_dec,thetas=pickle.load(f)
    elif iexp==2:
        with open(dat_dir_out+'/Loc_dec_4d_Foster15_3.pickle','rb') as f:
            loc_4d_dec,thetas=pickle.load(f)
    elif iexp==3:
        with open(dat_dir_out+'/Loc_dec_4d_Foster17_1.pickle','rb') as f:
            loc_4d_dec,thetas=pickle.load(f)
    elif iexp==4:
        with open(dat_dir_out+'/Loc_dec_4d_Bae21_1.pickle','rb') as f:
            loc_4d_dec,thetas=pickle.load(f)
    
    nsubs=len(loc_4d_dec)
    dec_bins=np.zeros((nsubs,len(ang_bins_temp)))

    for isub in range(nsubs):
        dec_temp=loc_4d_dec[isub]
        theta_temp=thetas[isub]

        # bin locations into current bin space
        temp=np.argmin(abs(circ_dist(ang_bins_temp,theta_temp,all_pairs=True)),axis=1)
        ang_bin=np.tile(ang_bins_temp,(len(theta_temp),1))        
        theta_bins=ang_bin[:,temp][0,:]

        for ibin,bin in enumerate(ang_bins_temp):
            dec_bins[isub,ibin]=np.mean(dec_temp[theta_bins==bin])
    
    if relative_change:
        dec_bins_mc=(dec_bins-np.mean(dec_bins,axis=1,keepdims=True))/np.mean(dec_bins,axis=1,keepdims=True)
        # dec_bins_mc=(dec_bins-np.max(dec_bins,axis=1,keepdims=True))/np.max(dec_bins,axis=1,keepdims=True)
    else:
        dec_bins_mc=dec_bins-np.mean(dec_bins,axis=1,keepdims=True)
    ang_bins=np.arange(0,2*np.pi,np.pi/8)
    ang_bins_deg=np.rad2deg(ang_bins)

    # smooth a little bit (just for consistency between experiments, barely changes anything)
    sigma=2/(180/len(ang_bins_deg))
    dec_bins_mc_smooth=np.concatenate((dec_bins_mc,dec_bins_mc,dec_bins_mc),axis=1)
    dec_bins_mc_smooth=gaussian_filter1d(dec_bins_mc_smooth,sigma=sigma,axis=1)
    dec_bins_mc_smooth=dec_bins_mc_smooth[:,dec_bins_mc.shape[1]:2*dec_bins_mc.shape[1]]

    # add first value to end to close the circle
    dec_bins_mc_smooth=np.concatenate((dec_bins_mc_smooth,dec_bins_mc_smooth[:,0:1]),axis=1)
    
    # add 360 to ang_bins_deg
    ang_bins_deg=np.concatenate((ang_bins_deg,[360]))
    
    if iexp==0:
        dec_bins_15_1_df=pd.DataFrame(dec_bins_mc_smooth)
    elif iexp==1:
        dec_bins_15_2_df=pd.DataFrame(dec_bins_mc_smooth)
    elif iexp==2:
        dec_bins_15_3_df=pd.DataFrame(dec_bins_mc_smooth)
    elif iexp==3:
        dec_bins_17_1_df=pd.DataFrame(dec_bins_mc_smooth)
    elif iexp==4:
        dec_bins_21_1_df=pd.DataFrame(dec_bins_mc_smooth)
        dec_bins_mc_smooth_Bae=np.copy(dec_bins_mc_smooth)

experiments=['Foster 2015, Exp. 1','Foster 2015, Exp. 2','Foster 2015, Exp. 3','Foster 2017, Exp. 1','Bae 2021, Exp. 1']

# combine all experiments into a single dataframe
dec_bins_df=pd.concat([dec_bins_15_1_df,dec_bins_15_2_df,dec_bins_15_3_df,dec_bins_17_1_df,dec_bins_21_1_df],axis=0)
dec_bins_mc_smooth=dec_bins_df.to_numpy()
dec_bins_df.columns=ang_bins_deg
dec_bins_df['subject']=np.arange(1,dec_bins_df.shape[0]+1)
dec_bins_df['Experiment']=np.concatenate((np.tile(experiments[0],dec_bins_15_1_df.shape[0]),np.tile(experiments[1],dec_bins_15_2_df.shape[0]),np.tile(experiments[2],dec_bins_15_3_df.shape[0]),np.tile(experiments[3],dec_bins_17_1_df.shape[0]),np.tile(experiments[4],dec_bins_21_1_df.shape[0])),axis=0)

# convert to long format
dec_bins_df=pd.melt(dec_bins_df,id_vars=['subject','Experiment'],var_name='orientation',value_name='decoding accuracy')
#% define vertical and horizontal bins for plotting
bin_width_deg=45
horz1_bins1=[0,0+bin_width_deg/2]
horz2_bins1=[360-bin_width_deg/2,360]
horz_bins2=[180-bin_width_deg/2,180+bin_width_deg/2]
vert_bins1=[90-bin_width_deg/2,90+bin_width_deg/2]
vert_bins2=[270-bin_width_deg/2,270+bin_width_deg/2]

angspace_vert1_ind=np.where((ang_bins_deg>vert_bins1[0]-.1)&(ang_bins_deg<vert_bins1[1]+.1))[0]
angspace_vert1=ang_bins_deg[angspace_vert1_ind]
angspace_vert2_ind=np.where((ang_bins_deg>vert_bins2[0]-.1)&(ang_bins_deg<vert_bins2[1]+.1))[0]
angspace_vert2=ang_bins_deg[angspace_vert2_ind]
angspace_horz1_ind1=np.where((ang_bins_deg<horz1_bins1[1]+.1))[0]
angspace_horz1_1=ang_bins_deg[angspace_horz1_ind1]
angspace_horz1_ind2=np.where((ang_bins_deg>horz2_bins1[0]-.1))[0]
angspace_horz1_2=ang_bins_deg[angspace_horz1_ind2]
angspace_horz2_ind=np.where((ang_bins_deg>horz_bins2[0]-.1)&(ang_bins_deg<horz_bins2[1]+.1))[0]
angspace_horz2=ang_bins_deg[angspace_horz2_ind]
#%% first let's plot the data by Bae 2021, Exp. 1

vert_color='purple'
horz_color='green'

if relative_change:
    dec_acc_ylim=[-.3,.3]
else:
    dec_acc_ylim=[-0.012,0.012]

n_boot=10000
plt.figure(figsize=(3.5,3))
plt.suptitle('Location Decoding, Bae 2021 Exp. 1')
# plt.axes(ax[0])

sns.lineplot(x='orientation',y='decoding accuracy',data=dec_bins_df[dec_bins_df['Experiment']==experiments[4]],errorbar=None,color='c',linewidth=2)

plt.axvline(90,color='k',linestyle='--',linewidth=1)
plt.axvline(180,color='k',linestyle='--',linewidth=1)
plt.axvline(270,color='k',linestyle='--',linewidth=1)
# plot horizontal line at 0
plt.axhline(0,color='k',linestyle='--',linewidth=1)

if relative_change:
    plt.yticks(np.arange(-.3,.301,.1))
else:
    plt.yticks(np.arange(-.01,.011,.005))

plt.xlim([0,360])
plt.ylim(dec_acc_ylim)
plt.xticks(np.arange(0,361,45),rotation=-45)
plt.xlabel('Location (deg)')
plt.ylabel('Relative decoding accuracy')

#%% now let's plot the aggregated data from ALL experiments
vert_color='purple'
horz_color='green'

if relative_change:
    dec_acc_ylim=[-.3,.3]
else:
    dec_acc_ylim=[-0.012,0.012]

n_boot=10000
plt.figure(figsize=(3.5,3))
plt.suptitle('Location Decoding, Aggregate')

sns.lineplot(x='orientation',y='decoding accuracy',data=dec_bins_df,errorbar=('ci',95),n_boot=n_boot,color='k',linewidth=2)
plt.axvline(90,color='k',linestyle='--',linewidth=1)
plt.axvline(180,color='k',linestyle='--',linewidth=1)
plt.axvline(270,color='k',linestyle='--',linewidth=1)
# plot horizontal line at 0
plt.axhline(0,color='k',linestyle='--',linewidth=1)

if relative_change:
    plt.yticks(np.arange(-.3,.301,.1))
else:
    plt.yticks(np.arange(-.01,.011,.005))

# make shaded area for vertical orientations
plt.fill_between(angspace_vert1,dec_acc_ylim[0],np.mean(dec_bins_mc_smooth[:,angspace_vert1_ind],axis=0), alpha=0.2, color=vert_color,zorder=0)
plt.fill_between(angspace_vert2,dec_acc_ylim[0],np.mean(dec_bins_mc_smooth[:,angspace_vert2_ind],axis=0), alpha=0.2, color=vert_color,zorder=0)

# make shaded area for horizontal orientations
plt.fill_between(angspace_horz1_1,dec_acc_ylim[0],np.mean(dec_bins_mc_smooth[:,angspace_horz1_ind1],axis=0), alpha=0.2, color=horz_color,zorder=0)
plt.fill_between(angspace_horz1_2,dec_acc_ylim[0],np.mean(dec_bins_mc_smooth[:,angspace_horz1_ind2],axis=0), alpha=0.2, color=horz_color,zorder=0)
plt.fill_between(angspace_horz2,dec_acc_ylim[0],np.mean(dec_bins_mc_smooth[:,angspace_horz2_ind],axis=0), alpha=0.2, color=horz_color,zorder=0)

plt.xlim([0,360])
plt.ylim(dec_acc_ylim)
plt.xticks(np.arange(0,361,45),rotation=-45)
plt.xlabel('Location (deg)')
plt.ylabel('Relative decoding accuracy')
plt.show()
#%%
if relative_change:
    vmin=-.15
    vmax=.2
else:
    vmin=-0.006
    vmax=0.008

# Generate a figure with a polar projection
fg = plt.figure(figsize=(8,8))
ax = fg.add_axes([0.1,0.1,0.8,0.8], projection='polar')

# Define colormap normalization for 0 to 2*pi
norm = mpl.colors.Normalize(0, 2*np.pi) 
# Plot a color mesh on the polar plot
# with the color set by the angle
dec_bins_mc=np.mean(dec_bins_mc_smooth_Foster,axis=0)
# put the first value at the end to close the circle
dec_bins_mc=np.concatenate((dec_bins_mc,[dec_bins_mc[0]]))
n = 129  #the number of secants for the mesh
t = np.linspace(0,2*np.pi,n)   #theta values
# r = np.linspace(.835,1,2)        #radius values change 0.6 to 0 for full circle
r = np.linspace(0,1,2)    
rg, tg = np.meshgrid(r,t)      #create a r,theta meshgrid
c = np.tile(dec_bins_mc,(2,1)).T                         #define color values as theta value
im = ax.pcolormesh(t, r, c.T,vmin=-.15,vmax=.20)  #plot the colormesh on axis with colormap
ax.set_yticklabels([])                   #turn of radial tick labels (yticks)
ax.tick_params(pad=15,labelsize=24)      #cosmetic changes to tick labels
ax.spines['polar'].set_visible(False)    #turn off the axis spine.
ax.set_xticklabels([])                   #turn of angular tick labels (xticks)
ax.set_yticklabels([])                   #turn of radial tick labels (yticks)
ax.set_xticks([])                       #turn of angular tick labels (xticks)
ax.set_yticks([])                       #turn of radial tick labels (yticks)

im.set_cmap('inferno')
# show the colorbar
fg.colorbar(im, ax=ax, orientation='vertical', pad=0.1, shrink=0.8)
plt.title('Foster et al. 2015, 2017',fontsize=24)
plt.show()

#%%
# Generate a figure with a polar projection
fg = plt.figure(figsize=(8,8))
ax = fg.add_axes([0.1,0.1,0.8,0.8], projection='polar')

# Define colormap normalization for 0 to 2*pi
norm = mpl.colors.Normalize(0, 2*np.pi) 

# Plot a color mesh on the polar plot
# with the color set by the angle
dec_bins_mc=np.mean(dec_bins_mc_smooth_Bae,axis=0)

n = 17  #the number of secants for the mesh
t = np.linspace(0,2*np.pi,n)   #theta values
r = np.linspace(0,1,2)    
rg, tg = np.meshgrid(r,t)      #create a r,theta meshgrid
c = np.tile(dec_bins_mc,(2,1)).T                         #define color values as theta value
im = ax.pcolormesh(t, r, c.T,vmin=vmin,vmax=vmax)  #plot the colormesh on axis with colormap
ax.set_yticklabels([])                   #turn of radial tick labels (yticks)
ax.tick_params(pad=15,labelsize=24)      #cosmetic changes to tick labels
ax.spines['polar'].set_visible(False)    #turn off the axis spine.
ax.set_xticklabels([])                   #turn of angular tick labels (xticks)
ax.set_yticklabels([])                   #turn of radial tick labels (yticks)
ax.set_xticks([])                       #turn of angular tick labels (xticks)
ax.set_yticks([])                       #turn of radial tick labels (yticks)

im.set_cmap('inferno')
# show the colorbar
fg.colorbar(im, ax=ax, orientation='vertical', pad=0.1, shrink=0.8)
plt.title('Bae 2021 Exp. 1',fontsize=24)
plt.show() 


