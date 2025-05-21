
# This script produces Fig 3b of the paper "Model mimicry limits conclusions about neural tuning and can mistakenly imply unlikely priors"
# It generates the output of the perfect cube model as describe in "Orientation Decoding in Human Visual Cortex: New Insights from an Unbiased Perspective" (2014)

import numpy as np
import matplotlib.pyplot as plt
import cv2 # need to install opencv-python

fig_out='/cs/home/wolffmj/Wolff2023/Cardinal_SNR_perception/figures'

#%% some local functions
def create_circular_mask(h, w, center=None, radius=None):

    import numpy as np
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = (dist_from_center-radius).clip(0,1)
    mask=1-mask
    return mask
def generate_gabor(shape, theta, phase, freq, sigma):
    """Generates a 2D sinewave grating multiplied with a Gaussian kernel. 

    Args:
        shape (_type_): imagesize in pixels
        theta (_type_): orientation in radians
        phase (_type_): phase (0 - np.pi)
        freq (_type_): spatial frequency (cycles per image)
        sigma (_type_): std of gaussian kernel 
        rng (_type_, optional): Defaults to np.random.

    Returns:
        gabors: size identical to input shape
    """
    
    assert isinstance(shape, tuple) and len(shape) == 2
    
    sigma_xs = np.copy(sigma)
    sigma_ys = np.copy(sigma)
    
    n_stds = int(shape[0]/2)
    # freq = freq/(n_stds*2)

    x, y = np.linspace(-n_stds, n_stds, shape[1]), np.linspace(-n_stds, n_stds, shape[0])
    X, Y = np.meshgrid(x, y)

    c, s = np.cos(theta), np.sin(theta)
    X1 = X * c + Y * s
    Y1 = -X * s + Y * c

    gabors = np.exp(-0.5 * (np.divide(X1, sigma_xs) ** 2 + np.divide(Y1,sigma_ys) ** 2))
    gabors *= np.cos((2 * np.pi) * freq * X1 + phase)

    return gabors
#%%
size_mult=1 # stimulus size multiplier

p_size=int(640*size_mult) # picture size in pixels
theta_grating=0 # orientation

n_phases=64 # how often dow we want to shift the phase and get a new image
# phases=np.linspace(1,360,n_phases)

phases=(np.arange(0,360,360/n_phases)) 

convolved=np.zeros((p_size,p_size,n_phases))

gabor_filter_orientations=np.linspace(0,180,9)
gabor_filter_orientations=np.delete(gabor_filter_orientations,-1)

# convolved_all_sum=np.empty((p_size,p_size,len(gabor_filter_orientations)))
convolved_all=np.empty((p_size,p_size,len(gabor_filter_orientations)))

filter_size=120*size_mult
filter_sd=20*size_mult
freq=0.025/size_mult
filter_phase=-90

# make masks for the circular grating, including the fixcation dot
mask_dot=create_circular_mask(p_size,p_size,radius=12*size_mult)
mask_dot=-mask_dot+1

mask=create_circular_mask(p_size,p_size,radius=237.5*size_mult)
#%%
for iorientation, orientation in enumerate(gabor_filter_orientations):

    convolved=np.zeros((p_size,p_size,n_phases))

    print(orientation)
    gabor_filter=generate_gabor((filter_size,filter_size), np.deg2rad(orientation), np.deg2rad(filter_phase), freq, filter_sd)
    masked_filter = np.copy(gabor_filter)
    # pad to p_size
    masked_filter=np.pad(masked_filter,((int((p_size-filter_size)/2),int((p_size-filter_size)/2)),(int((p_size-filter_size)/2),int((p_size-filter_size)/2))),'constant',constant_values=0)

    for iphase, phase in enumerate(phases):
        # print(iphase)
        grating=generate_gabor((p_size,p_size), np.deg2rad(theta_grating), np.deg2rad(phase), freq, float('inf'))
        masked_grating = np.copy(grating)
        masked_grating=masked_grating*mask
        masked_grating=masked_grating*mask_dot

        # rescale to 0 to 255
        masked_grating=masked_grating+1
        masked_grating=masked_grating*127.5

        # convolve the grating with the gabor filter
        convolved_temp=cv2.filter2D(masked_grating,-1,gabor_filter)
        convolved_temp=convolved_temp**2

        convolved[:,:,iphase] = convolved_temp

    convolved_all[:,:,iorientation] = np.mean(convolved,axis=-1)/np.max(np.mean(convolved,axis=-1))

convolved_sum=np.sum(convolved_all,axis=-1)

grating_vert=np.copy(masked_grating)
grating_horz=np.rot90(grating_vert)

model_vert=convolved_sum
model_horz=np.rot90(convolved_sum)

#%% 
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].imshow(grating_vert, cmap='gray')
axs[0, 0].axis('off')
axs[1, 0].imshow(grating_horz, cmap='gray')
axs[1, 0].axis('off')
axs[0, 1].imshow(model_vert, cmap='gray')
axs[0, 1].axis('off')
axs[1, 1].imshow(model_horz, cmap='gray')
axs[1, 1].axis('off')
plt.show()
#%%
# create source data by saving this to an excel file
source_dat_out='/cs/home/wolffmj/Wolff2023/Cardinal_SNR_perception/upload_folder/Source_data'
import pandas as pd
df_grating_vert = pd.DataFrame(grating_vert)
df_grating_horz = pd.DataFrame(grating_horz)
df_model_vert = pd.DataFrame(model_vert)
df_model_horz = pd.DataFrame(model_horz)
with pd.ExcelWriter(source_dat_out + '/Fig3b_source_data.xlsx') as writer:
    df_grating_vert.to_excel(writer, sheet_name='Fig3b vertical grating', index=False)
    df_grating_horz.to_excel(writer, sheet_name='Fig3b horizontal grating', index=False)
    df_model_vert.to_excel(writer, sheet_name='Fig3b model output vertical', index=False)
    df_model_horz.to_excel(writer, sheet_name='Fig3b model output horizontal', index=False)
