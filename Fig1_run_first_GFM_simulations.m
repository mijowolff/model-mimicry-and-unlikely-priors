% This script produces the simulation data related to Figure 1 in Wolff and
% Rademaker (2024)
% The output of this is loaded into Fig1_run_second_GFM_decoding.py
% precomputed simulation data obtained from this scripts can be found at https://osf.io/bdf74/

% this script uses ALOT of code and code snippets published by Harrison et al. 2023 et al.
% available at https://github.com/ReubenRideaux/Neural-tuning-instantiates-prior-expectations-in-the-human-visual-system

clear all

out_folder=''; % save simulated data here

% define functions
vmfun = @(xx, kappa, mu) exp(kappa*cosd((xx-mu))); % von Mises function
csfun = @(xx, mu, expo) (cosd(xx-mu)).^expo; % rectified cosine function
dgfun = @(xx, kappa, mu)    diff(exp( kappa*(cosd(xx-mu)-1) )); % differential von Mises function

% define paramters
params.meta.n_subs = 36; % # of particiapnts to simulate (scalar, integer)
params.tcurves.n_ori_prefs  = 16; % # of orientation preferences (scalar, integer)
params.tcurves.n_oris = 180; % # of orientations to test (deg; scalar, integer)
params.tcurves.oris = linspace(1,180,params.tcurves.n_oris); % (deg)
params.tcurves.ori_prefs = linspace(0,180,params.tcurves.n_ori_prefs+1); % (deg)
params.tcurves.ori_prefs(end) = [];

params.tcurves.kappa = 2; % neural tuning kappa value (deg)

params.cardi.oris = [0,90]; % position of cardinal orientations (deg)
params.cardi.sft.kappa = .5; % bias tuning kappa value (deg)

params.neural.n_sensors = 32; % # of EEG sensors to simulate (scalar, integer)
params.neural.sensornoise_sd = 6; % amplitude of sensor noise (a.u.; scalar, float)

params.stim.n_stimuli = 180; % # of different oriented grating to simulate (scalar, integer)
params.stim.n_repeats = 18*2; % # of repeats per grating (scalar, integer)
params.stim.n_trials = params.stim.n_stimuli * params.stim.n_repeats; % total # of trials to simulate
params.stim.stim_lib = repmat(1:params.stim.n_stimuli,1,params.stim.n_repeats);

params.model.n_ori_chans = 6; % # of forward model channels (scalar, integer)
params.model.kernel = exp(1i * (params.tcurves.oris*2*pi/180)); % circular kernel

params.analysis.smoothing = 16; % smoothing window
%%
% simulate neuron tuning curves
upsample_factor = 1000000;
upsampled_oris = linspace(0,180,upsample_factor+1);
%% make preferred channel tuning model
params.cardi.sft.amp_pref = [14,8]; % amplitude of bias at each cardinal (a.u.)
params.cardi.sft.mod_pref  = dgfun(upsampled_oris*2,params.cardi.sft.kappa,params.cardi.oris(1)*2)*params.cardi.sft.amp_pref (1)+...
    dgfun(upsampled_oris*2,params.cardi.sft.kappa,params.cardi.oris(2)*2)*params.cardi.sft.amp_pref (2);

params.cardi.sft.mod_pref = params.cardi.sft.mod_pref(round(linspace(1,upsample_factor,params.tcurves.n_ori_prefs+1)));
params.cardi.sft.mod_pref(end) = [];

params.cardi.sft.mod_pref = params.cardi.sft.mod_pref/max(abs(params.cardi.sft.mod_pref))*max(abs(params.cardi.sft.amp_pref));
params.neural.sensornoise_sd_pref = 6; % amplitude of sensor noise (a.u.; scalar, float)

tcurves_pref=zeros(params.stim.n_stimuli,params.tcurves.n_ori_prefs); 
for cc = 1:params.tcurves.n_ori_prefs      
    tcurves_pref(:,cc) = vmfun(params.tcurves.oris*2,params.tcurves.kappa,params.tcurves.ori_prefs(cc)*2+params.cardi.sft.mod_pref(cc));
    tcurves_pref(:,cc) = tcurves_pref(:,cc)/max(tcurves_pref(:,cc));   
end
%% make gain tuning model
params.cardi.sft.amp_gain = [15,8]; % amplitude of bias at each cardinal (a.u.)
params.cardi.sft.min_max_gain =[1.4,.7]; % minmum/maximum channel response amplitudes

temp1=vmfun(upsampled_oris*2,params.tcurves.kappa,params.cardi.oris(1)*2)*params.cardi.sft.amp_gain(1);
temp2=vmfun(upsampled_oris*2,params.tcurves.kappa,params.cardi.oris(2)*2)*params.cardi.sft.amp_gain(2);
gain_mod=((temp1+temp2)/max(temp1+temp2));

gain_mod=gain_mod(round(linspace(1,upsample_factor,params.tcurves.n_ori_prefs+1)));
gain_mod(end)=[];
ori_prefs=params.tcurves.ori_prefs;

if isnan(params.cardi.sft.min_max_gain)
    gain_mod=gain_mod*(1/mean(gain_mod));
end

if ~isnan(params.cardi.sft.min_max_gain)
    bg=params.cardi.sft.min_max_gain(1);
    ag=params.cardi.sft.min_max_gain(2);
    gain_mod=(bg - ag) * (gain_mod - min(gain_mod)) / (max(gain_mod) - min(gain_mod)) + ag;
end

params.cardi.sft.mod_gain=gain_mod;

tcurves_gain=zeros(params.stim.n_stimuli,params.tcurves.n_ori_prefs); 
for cc = 1:params.tcurves.n_ori_prefs      
    tcurves_gain(:,cc) = vmfun(params.tcurves.oris*2,params.tcurves.kappa,ori_prefs(cc)*2);
    tcurves_gain(:,cc) = (tcurves_gain(:,cc)/max(tcurves_gain(:,cc)))*params.cardi.sft.mod_gain(cc);    
end
%% make uniform with SNR tuning model
params.cardi.sft.min_max_uni =[1,.68]; % minium and maximum signal strenght change (used after converting to sensor space)

params.cardi.signal_strengths_uni = [15,8]; % amplitudes of signal strength change at each cardinal
temp1=vmfun(params.tcurves.oris*2,params.tcurves.kappa,params.cardi.oris(1)*2)*params.cardi.signal_strengths_uni(1);
temp2=vmfun(params.tcurves.oris*2,params.tcurves.kappa,params.cardi.oris(2)*2)*params.cardi.signal_strengths_uni(2);

bs=params.cardi.sft.min_max_uni(1);
as=params.cardi.sft.min_max_uni(2);

signal_levels=(temp1+temp2)';

signal_levels=(bs - as) * (signal_levels - min(signal_levels)) / (max(signal_levels) - min(signal_levels)) + as;
params.cardi.signal_strengths_uni=signal_levels;

tcurves_uni=zeros(params.stim.n_stimuli,params.tcurves.n_ori_prefs); 
for cc = 1:params.tcurves.n_ori_prefs      
    tcurves_uni(:,cc) = vmfun(params.tcurves.oris*2,params.tcurves.kappa,ori_prefs(cc)*2);
    tcurves_uni(:,cc) = (tcurves_gain(:,cc)/max(tcurves_gain(:,cc)));    
end
%% make width tuning model
params.tcurves.n_ori_prefs_width  = 24; % # of orientation preferences (scalar, integer)
params.tcurves.n_oris_width = 180; % # of orientations to test (deg; scalar, integer)
params.tcurves.oris_width = linspace(1,180,params.tcurves.n_oris_width); % (deg)
params.tcurves.ori_prefs_width = linspace(0,180,params.tcurves.n_ori_prefs_width+1); % (deg)
params.tcurves.ori_prefs_width(end) = [];

params.cardi.sft.amp_width = [15,4]; % amplitude of bias at each cardinal (a.u.)
params.cardi.sft.min_max_width =[7,19]; % scale to maximum minimum widths
params.cardi.sft.sign_change_width=1.5; % change achannel response amplitudes
bw=params.cardi.sft.min_max_width(1);
aw=params.cardi.sft.min_max_width(2);

temp1=vmfun(upsampled_oris*2,params.tcurves.kappa,params.cardi.oris(1)*2)*params.cardi.sft.amp_width(1);
temp2=vmfun(upsampled_oris*2,params.tcurves.kappa,params.cardi.oris(2)*2)*params.cardi.sft.amp_width(2);
width_mod=((temp1+temp2)/max(temp1+temp2));
width_mod=width_mod(round(linspace(1,upsample_factor,params.tcurves.n_ori_prefs_width+1)));
width_mod(end)=[];
width_mod=(bw - aw) * (width_mod - min(width_mod)) / (max(width_mod) - min(width_mod)) + aw;
params.cardi.sft.mod_width=width_mod;

tcurves_width=zeros(params.stim.n_stimuli,params.tcurves.n_ori_prefs_width); 
for cc = 1:params.tcurves.n_ori_prefs_width
       
    tcurves_width(:,cc)=(vmfun(params.tcurves.oris_width*2,params.cardi.sft.mod_width(cc),params.tcurves.ori_prefs_width(cc)*2));
    tcurves_width(:,cc) = (tcurves_width(:,cc)/max(tcurves_width(:,cc)))*params.cardi.sft.sign_change_width;   
end
%% begin simulation

warning OFF
for sub = 1:params.meta.n_subs % repeat over multiple subjects

    neuron_to_sensor_weights = rand(params.tcurves.n_ori_prefs,params.neural.n_sensors);
    neuron_to_sensor_weights_width = rand(params.tcurves.n_ori_prefs_width,params.neural.n_sensors);

    stimuli = (params.stim.stim_lib);
    
    sensornoise=(randn(params.stim.n_trials,params.neural.n_sensors))*params.neural.sensornoise_sd;
    
    % compute the sensor responses
    sensor_response_pref = tcurves_pref(stimuli,:)*neuron_to_sensor_weights + sensornoise ;
    sensor_response_gain = tcurves_gain(stimuli,:)*neuron_to_sensor_weights + sensornoise;
    sensor_response_width = tcurves_width(stimuli,:)*neuron_to_sensor_weights_width + sensornoise;
    
    sensor_response_uni = (tcurves_uni(stimuli,:))*neuron_to_sensor_weights;
    sensor_response_uni=(sensor_response_uni.*params.cardi.signal_strengths_uni(stimuli))+sensornoise;
    
    stimuli_subs(sub,:)=stimuli;
%     
    sensor_response_pref_subs(sub,:,:)=sensor_response_pref;
    sensor_response_gain_subs(sub,:,:)=sensor_response_gain;
    sensor_response_width_subs(sub,:,:)=sensor_response_width;
    sensor_response_uni_subs(sub,:,:)=sensor_response_uni;
    
end
save(fullfile(out_folder,'GFM_simulation_data.mat'),'stimuli_subs','sensor_response_gain_subs','sensor_response_pref_subs',...
    'sensor_response_width_subs','sensor_response_uni_subs','params','signal_levels','tcurves_pref','tcurves_gain','tcurves_width','tcurves_uni');
