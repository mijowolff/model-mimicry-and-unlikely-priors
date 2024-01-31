
# all cusomt functions used in analyses for the manuscript:
# "Model mimicry prohibits  limit conclusions about neural tuning and can mistakenly imply unlikely priors"
# by: Michael J. Wolff & Rosanne L. Rademaker, 2024

import numpy as np
from scipy.ndimage.filters import uniform_filter1d
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy.spatial import distance
from sklearn.covariance import LedoitWolf,OAS,ShrunkCovariance,EmpiricalCovariance,MinCovDet
from numpy.linalg import pinv,inv
import random

#%%
def circ_mean(alpha, axis=None, w=None):
    if w is None:
        w = np.ones(alpha.shape)

    # weight sum of cos & sin angles
    t = w * np.exp(1j * alpha)

    r = np.sum(t, axis=axis)
    
    mu = np.angle(r)

    return mu
#%%
def circ_dist(x,y,all_pairs=False):
    
    # circular distance between angles in radians
    
    x=np.asarray(x)
    y=np.asarray(y)
    
    x=np.squeeze(x)
    y=np.squeeze(y)
    
    if all_pairs:
        x_new=np.tile(np.exp(1j*x),(len(y),1))
        y_new=np.transpose(np.tile(np.exp(1j*y),(len(x),1)))
        circ_dists= np.angle(x_new/y_new)
    else:
        circ_dists= np.angle(np.exp(1j*x)/np.exp(1j*y))
        
    return circ_dists
#%%
def covdiag(x):
    
    '''
    x (t*n): t iid observations on n random variables
    sigma (n*n): invertible covariance matrix estimator
    
    Shrinks towards diagonal matrix
    as described in Ledoit and Wolf, 2004
    '''
    
    t,n=np.shape(x)
    
    # de-mean
    x=x-np.mean(x,axis=0)
    
    #get sample covariance matrix
    sample=np.cov(x,rowvar=False,bias=True)
    
    #compute prior
    prior=np.zeros((n,n))
    np.fill_diagonal(prior,np.diag(sample))
    
    #compute shrinkage parameters
    d=1/n*np.linalg.norm(sample-prior,ord='fro')**2
    y=x**2
    r2=1/n/t**2*np.sum(np.dot(y.T,y))-1/n/t*np.sum(sample**2)
    
    #compute the estimator
    shrinkage=max(0,min(1,r2/d))
    sigma=shrinkage*prior+(1-shrinkage)*sample
    
    return sigma
#%%
# bunch of different ways to compute the convariance matrix
# I always use the inverse of covdiag
def inverse_cov_fun(data,cov_metric,inverse_method='inv'):
    
    if cov_metric=='covdiag':
        cov=covdiag(data)
    if cov_metric=='LedoitWolf' or cov_metric=='LW' or cov_metric=='lw':
        cov_temp = LedoitWolf().fit(data)
        cov=cov_temp.covariance_
    elif cov_metric=='OAS':
        cov_temp = OAS().fit(data)
        cov=cov_temp.covariance_
    elif cov_metric=='Empircial' or cov_metric=='empirical' or cov_metric=='EmpiricalCovariance' or cov_metric=='empirical_covariance':
        cov_temp=EmpiricalCovariance().fit(data)
        cov=cov_temp.covariance_
    elif cov_metric=='Shrunk' or cov_metric=='shrunk' or cov_metric=='ShrunkCovariance':
        cov_temp=ShrunkCovariance().fit(data)
        cov=cov_temp.covariance_
    elif cov_metric=='MinCovDet':
        cov_temp=MinCovDet().fit(data)
        cov=cov_temp.covariance_
    elif cov_metric=='normal':
        cov=np.cov(np.transpose(data-np.mean(data)))
    else:
        cov_temp = LedoitWolf().fit(data)
        cov=cov_temp.covariance_
    if inverse_method=='pinv':
        cov_inv=pinv(cov)
    elif inverse_method=='inv':
        cov_inv=inv(cov)
    else:
        cov_inv=cov
    return cov_inv 
#%%
def cosfun(theta,mu,basis_smooth,amplitude='default',offset='default'):

    if amplitude=='default':
        amplitude=.5
    if offset=='default':
        offset=.5
    return (offset+amplitude*np.cos(theta-mu))**basis_smooth
#%%
def basis_set_fun(theta_bins,u_theta,basis_smooth='default'):
        
    if basis_smooth=='default':
        basis_smooth=theta_bins.shape[0]-1
        
    smooth_bins=np.zeros(theta_bins.shape)
    
    for ci in range(theta_bins.shape[0]):
        temp_kernel=cosfun(u_theta,u_theta[ci],basis_smooth)
        temp_kernel=np.expand_dims(temp_kernel,axis=[1,2])
        temp_kernel=np.tile(temp_kernel,(1,theta_bins.shape[1],theta_bins.shape[2]))
        smooth_bins[ci,:,:]=np.sum(theta_bins*temp_kernel,axis=0)/sum(temp_kernel)                        
    
    return smooth_bins

#%%
# combine channel and time dimensions 
def dat_prep_4d_section(data,time_dat=None,toi=None,span=10,hz=500,relative_baseline=True,in_ms=True):    
    
    if toi is not None:
        if time_dat is not None:
            hz=float(np.round(1/np.diff(time_dat[:2]))) # determine sample-rate of input data
            time_dat=np.squeeze(time_dat)
            toi_new=np.zeros(2)
            toi_new[1]=toi[1]+(1000/hz)/4000 # just in case, to avoid possibly missing a time-point...
            toi_new[0]=toi[0]
            data=data[:,:,(time_dat>toi_new[0])&(time_dat<toi_new[1])]           
    
    if relative_baseline:
        w_mean=np.mean(data,axis=-1,keepdims=True)
        data=data-np.tile(w_mean,(1,1,data.shape[-1])) 
    
    if time_dat is not None:
        hz=float(np.round(1/np.diff(time_dat[:2])))
    
    if in_ms:         
        span=int(span/(1000/hz))
    
    data2=uniform_filter1d(data,span,axis=-1,mode='constant')
    
    data3=data2[:,:,int(np.floor(span/2))::span] 

    dat_dec=data3.reshape(data3.shape[0],data3.shape[1]*data3.shape[2],order='F')

    return dat_dec

#%%
#%%  orientation resconstrution using cross-validation
def dist_theta_kfold(data,theta,n_folds=8,n_reps=10,data_trn=None,basis_set=True,cov_metric='covdiag',cov_tp=True,angspace='default',ang_steps=4,balanced_train_bins=True,balanced_cov=False,residual_cov=False,dist_metric='mahalanobis',verbose=True):
    
    if verbose:
        from progress.bar import ChargingBar

    if data_trn is None:
        data_trn=data
        
    if type(angspace)==str:
        if angspace=='default':
            angspace=np.arange(-np.pi,np.pi,np.pi/8) # default is 16 bins
        
    if np.array_equal(angspace,np.unique(theta)):
        ang_steps=1        
                
    bin_width=np.diff(angspace)[0]
    
    x_dummy=np.zeros(len(theta)) # needed for sklearn splitting function
    
    X_ts=data
    X_tr=data_trn    
    if len(X_tr.shape)<3:
        X_tr=np.expand_dims(X_tr,axis=-1)
        
    if len(X_ts.shape)<3:
        X_ts=np.expand_dims(X_ts,axis=-1)
            
    ntrls, nchans, ntps=np.shape(X_ts)  

    m_temp=np.zeros((len(angspace),nchans,ntps))
    m=m_temp
    
    if dist_metric=='euclidean':
        cov_metric=False 
    
    if verbose:
        bar = ChargingBar('Processing', max=ntps*ang_steps*n_reps*n_folds)
    
    distances=np.empty((ang_steps,len(angspace),ntrls,ntps))
    
    distances[:]=np.NaN

    angspaces=np.zeros((ang_steps,len(angspace)))

    for ans in range(0,ang_steps): # loop over all desired orientation spaces
    
        angspace_temp=angspace+ans*bin_width/ang_steps
        angspaces[ans,:]=angspace_temp

    angspace_full=np.reshape(angspaces,(angspaces.shape[0]*angspaces.shape[1]),order='F')

    theta_dists=circ_dist(angspace_full,theta,all_pairs=True)
    theta_dists=theta_dists.transpose()  

    theta_dists_temp=np.expand_dims(theta_dists,axis=-1)
    theta_dists2=np.tile(theta_dists_temp,(1,1,ntps))

    for ans in range(0,ang_steps): # loop over all desired orientation spaces
    
        angspace_temp=angspace+ans*bin_width/ang_steps
        
        # convert orientations into bins
        temp=np.argmin(abs(circ_dist(angspace_temp,theta,all_pairs=True)),axis=1)
        ang_bin_temp=np.tile(angspace_temp,(len(theta),1))               
        bin_orient_rads=ang_bin_temp[:,temp][0,:]
        
        y_subst=temp
        y=bin_orient_rads
                
        rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_reps) # get splitting object
        
        split_counter=0
        
        distances_temp=np.empty([len(angspace_temp),ntrls,n_reps,ntps])
        distances_temp[:]=np.NaN
               
        for train_index, test_index in rskf.split(X=x_dummy,y=y_subst): # loop over all train/test folds, and repepitions
            
            X_train, X_test = X_tr[train_index,:,:], X_ts[test_index,:,:]
            y_train, y_test = y[train_index], y[test_index]
            y_subst_train, y_subst_test = y_subst[train_index], y_subst[test_index]
                        
            irep=int(np.floor(split_counter/n_folds))
            split_counter=split_counter+1
          
            train_dat_cov = np.empty((0,X_train.shape[1],X_train.shape[2]))
            train_dat_cov[:]=np.NaN
            
            if balanced_train_bins: # average over same orientaions of training set, but make sure these averages are based on balanced trials
                count_min=min(np.bincount(y_subst_train))
                for c in range(len(angspace_temp)):
                    temp_dat=X_train[y_train==angspace_temp[c],:,:]
                    ind=random.sample(list(range(temp_dat.shape[0])),count_min)
                    m_temp[c,:,:]=np.mean(temp_dat[ind,:,:],axis=0)
                    if balanced_cov: # if desired, the data used for the covariance can also be balanced
                        if residual_cov: # take the residual, note that this should only be done if the cov data is balanced!
                            train_dat_cov = np.append(train_dat_cov, temp_dat[ind,:,:]-np.mean(temp_dat[ind,:,:],axis=0), axis=0)
                        else:
                            train_dat_cov = np.append(train_dat_cov, temp_dat[ind,:,:], axis=0)
            else:
                for c in range(len(angspace_temp)):
                    m_temp[c,:,:]=np.mean(X_train[y_train==angspace_temp[c],:,:],axis=0)
                    
            if basis_set: # smooth the averaged train data with basis set
                m=basis_set_fun(m_temp,angspace_temp,basis_smooth='default')
            else:
                m=m_temp
            
            if not balanced_cov:
                train_dat_cov=X_train # use all train trials if cov is not balanced
            
            if cov_metric and not cov_tp: # can compute covariance once of the average of all time-points (can speed things up)
                train_dat_cov=np.mean(train_dat_cov,axis=-1,keepdims=False) 
                cov=inverse_cov_fun(train_dat_cov,cov_metric,inverse_method='inv')                      
                    
            for tp in range(ntps):
                m_train_tp=m[:,:,tp]
                X_test_tp=X_test[:,:,tp]
                
                if cov_metric:
                    if cov_tp: # get separate covariance matrix for each time-point separately
                        dat_cov_tp=train_dat_cov[:,:,tp]
                        cov=inverse_cov_fun(dat_cov_tp,cov_metric,inverse_method='inv')   
                    distances_temp[:,test_index,irep,tp]=distance.cdist(m_train_tp,X_test_tp,'mahalanobis', VI=cov) # compute distances between all test trials, and average train trials
                else:                    
                    distances_temp[:,test_index,irep,tp]=distance.cdist(m_train_tp,X_test_tp,'euclidean')
                   
                if verbose:    
                    bar.next()

        distances[ans,:,:,:]=np.mean(distances_temp,axis=2,keepdims=False)
    
    distances=distances-np.mean(distances,axis=1,keepdims=True)
    distances_flat=np.reshape(distances,(distances.shape[0]*distances.shape[1],distances.shape[2],distances.shape[3]),order='F')
    distances_flat=distances_flat-np.mean(distances_flat,axis=0,keepdims=True)
    # dec_cos=-np.mean(np.mean(np.cos(theta_dists2)*distances_flat,axis=0),axis=1)
    dec_cos=np.squeeze(-np.mean(np.cos(theta_dists2)*distances_flat,axis=0))

    # order the distances, such that same angle distances are in the middle
    # first, assign each theta to a bin from angspace_full
    temp=np.argmin(abs(circ_dist(angspace_full,theta,all_pairs=True)),axis=1)
    ang_bin_temp=np.tile(angspace_full,(len(theta),1))               
    theta_bins=ang_bin_temp[:,temp][0,:]

    # then, sort the distances based on the distances between the theta_bins
    theta_bin_dists=circ_dist(angspace_full,theta_bins,all_pairs=True)
    theta_bin_dists=theta_bin_dists.transpose()
    theta_bin_dists_abs=np.abs(theta_bin_dists)
    # get index of the minimum distance
    theta_bin_dists_min_ind=np.argmin(theta_bin_dists_abs,axis=0)

    distances_ordered=np.zeros((distances_flat.shape))

    shift_to=np.where(angspace_full==0)[0][0]
    for trl in range(len(theta)):
        distances_ordered[:,trl,:] = np.roll(distances_flat[:,trl,:], int(shift_to - theta_bin_dists_min_ind[trl]), axis=0)
    
    if verbose:
        bar.finish()
    
    return dec_cos,distances,distances_ordered,angspaces,angspace_full
