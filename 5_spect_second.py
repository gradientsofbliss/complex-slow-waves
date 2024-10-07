# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import scipy
warnings.simplefilter(action='ignore', category=FutureWarning)

# set font and figs
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['figure.dpi'] = 500
plt.rcParams['savefig.dpi'] = 500
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = "cm"
plt.rcParams.update({'font.size': 14})

import scipy.stats as stats
import pingouin as pg
import statsmodels.stats.multitest as multi

import emd
import mne

import os
print(os.getcwd())

full_perps = ['PCJ09', 'PDH47', 'PLM37', ' PVU29', 'PTS72', 'P2M63', 'P5P11', 'PBA32']
peak_perps = ['PCJ09', 'PFM82', 'PDH47', 'PLM37', 'PJS04', 'PVU29', 'PUA35', 'PTS72', 'P2M63', 'P5P11', 'PGH44', 'P4O85', 'PBA32']
seg_perps = ['PCJ09', 'PFM82', 'PDH47', 'PLM37', 'PJS04', 'PVU29', 'PUA35', 'PTS72', 'PXH23', 'PQS29', 'P2M63', 'P5P11', 'PGH44', 'P4O85', 'P3N74', 'PBA32', 'PDL71', 'P7R50', 'PFQ62']

perps = seg_perps

conditions = ['Rest', '5-MeO-DMT']
sfreq = 500
n_regions = 64

rest_full_epochs = np.arange(0,140)
five_full_epochs = np.arange(0,400)

good_epoch_dict = np.load('good_epoch_dict.npy', allow_pickle=True).item()
good_perp_epoch_dict = np.load('good_perp_epoch_dict.npy', allow_pickle=True).item()

lobes = ['frontal', 'central', 'parietal', 'temporal', 'occipital']
electrode_lobe_dict = np.load('electrode_lobe_dict.npy', allow_pickle=True).item()
electrode_list = electrode_lobe_dict['Frontal'] + electrode_lobe_dict['Central'] + electrode_lobe_dict['Parietal'] + electrode_lobe_dict['Temporal'] + electrode_lobe_dict['Occipital']
electrode_lobe_dict['Global'] = electrode_list
electrode_order = np.load('electrode_order.npy', allow_pickle=True)

band_range_dict = {'slow': [0.5,1.5], 'delta': [1.5,4], 'theta': [4,8], 'alpha': [8,12], 'beta': [12,30], 'gamma': [30,50]}
band_symbol_dict = {'slow': (r'$s$'), 'delta': (r'$\delta$'), 'theta': (r'$\theta$'), 'alpha': (r'$\alpha$'), 'beta': (r'$\beta$'), 'gamma': (r'$\gamma$')}

ts = ['peak', 'Ppeak']

def aug_t_test(array1, array2, paired=True):
    '''
    T-test

    Parameters
    ----------
    array1: array
    array2: array
    
    array1 first so positive T mean array1 > array2

    '''
    import pingouin as pg

    result = pg.ttest(array1, array2, paired=paired)
    
    t= result['T'].values[0]
    p= result['p-val'].values[0]
    cil= result['CI95%'].values[0][0]
    ciu= result['CI95%'].values[0][1]
    d= result['cohen-d'].values[0]
    bf= result['BF10'].values[0]
    dof= result['dof'].values[0]
    power= result['power'].values[0]

    return t, p, cil, ciu, d, bf, dof, power

def write_scientific(val):
    val = float(val)
    # make it x 10^y
    if abs(val) < 0.001 or abs(val) > 1000:
        val = f'{val:.2e}'
        val = val.split('e')
        x= f'{val[0]}$\\times10^{{{int(val[1])}}}$'
    else:
        x= f'{val:.3f}'
    return x

raw = mne.io.read_raw_edf('raw.edf', preload=True)
raw.rename_channels(lambda x: x[:-4])
raw.rename_channels(lambda x: x[4:])
ch_names = raw.info['ch_names']
for i in ch_names:
    if i == 'EOG':
        raw.set_channel_types(mapping={'EOG':'eog'})
    else:
        raw.set_channel_types(mapping={i:'eeg'})
montage = mne.channels.make_standard_montage('standard_1020')
ind = [i for (i, channel) in enumerate(montage.ch_names) if channel in ch_names]
montage.ch_names = [montage.ch_names[i] for i in ind]
kept_channel_info = [montage.dig[x+3] for x in ind]
montage.dig = montage.dig[0:3]+kept_channel_info
for i in montage.dig[3:]:
    i['r'][1] = i['r'][1] - 0.019
raw.set_montage(montage, match_case=True)

# %%
#######
# E ###
#######

# %%
res = 149
freq_edges, freq_centres = emd.spectra.define_hist_bins(0.5, 50, res, 'linear')
freq_centres = np.round(freq_centres, 1)

for condition in conditions:

    if not os.path.exists('Spectra/'+condition+'_emd_E_spectrogram.csv'):
        if condition == 'Rest':
            full = rest_full_epochs
        else:
            full = five_full_epochs
        df = pd.DataFrame(columns=full, index=freq_centres)
        emd_E_spectrogram_cond_avg_df = df.copy(); fft_E_spectrogram_cond_avg_df = df.copy()
        emd_E_spectrogram_cond_se_df = df.copy(); fft_E_spectrogram_cond_se_df = df.copy()
        for epoch in full:
            good_perps = good_perp_epoch_dict[condition, epoch]
            for perp in good_perps:
                emd_spectra_df = pd.read_csv('Spectra/'+perp+'/'+condition+'/'+perp+'_'+condition+'_E_emd_spectrogram.csv', index_col=0)
                fft_spectra_df = pd.read_csv('Spectra/'+perp+'/'+condition+'/'+perp+'_'+condition+'_E_fft_spectrogram.csv', index_col=0)
                emd_spect_epoch = np.array(emd_spectra_df[str(epoch)])
                fft_spect_epoch = np.array(fft_spectra_df[str(epoch)])
                if perp == good_perps[0]:
                    emd_spectra_cond_ext = emd_spect_epoch
                    fft_spectra_cond_ext = fft_spect_epoch
                else:
                    emd_spectra_cond_ext = np.vstack((emd_spectra_cond_ext, emd_spect_epoch))
                    fft_spectra_cond_ext = np.vstack((fft_spectra_cond_ext, fft_spect_epoch))
            emd_E_spectrogram_cond_avg_df[epoch] = np.nanmean(emd_spectra_cond_ext, axis=0)
            fft_E_spectrogram_cond_avg_df[epoch] = np.nanmean(fft_spectra_cond_ext, axis=0)
            emd_E_spectrogram_cond_se_df[epoch] = np.nanstd(emd_spectra_cond_ext, axis=0)/np.sqrt(len(good_perps))
            fft_E_spectrogram_cond_se_df[epoch] = np.nanstd(fft_spectra_cond_ext, axis=0)/np.sqrt(len(good_perps))
        emd_E_spectrogram_cond_avg_df.to_csv('Spectra/'+condition+'_emd_E_spectrogram.csv')
        fft_E_spectrogram_cond_avg_df.to_csv('Spectra/'+condition+'_fft_E_spectrogram.csv')
        emd_E_spectrogram_cond_se_df.to_csv('Spectra/'+condition+'_emd_E_spectrogram_se.csv')
        fft_E_spectrogram_cond_se_df.to_csv('Spectra/'+condition+'_fft_E_spectrogram_se.csv')

emd_spectra_cond_avg_df = pd.read_csv('Spectra/5-MeO-DMT_emd_E_spectrogram.csv', index_col=0)
fft_spectra_cond_avg_df = pd.read_csv('Spectra/5-MeO-DMT_fft_E_spectrogram.csv', index_col=0)

emd_spectra_cond_avg_df = np.log10(emd_spectra_cond_avg_df)
fft_spectra_cond_avg_df = np.log10(fft_spectra_cond_avg_df)

def plot_spectrogram(spectra_cond_avg_df):
    fig, ax = plt.subplots(1, 1, figsize=(8.5,5))
    import cmasher as cma
    cmap = cma.pride
    ax.imshow(spectra_cond_avg_df, cmap=cmap, aspect='auto')
    ax.invert_yaxis()
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xticks([0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400], ['0', '2', '4', '6', '8', '10', '12', '14', '16', '18', '20'], rotation=0)
    ax.set_yticks([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 148], ['0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50'], rotation=0)
    # colorbar
    ax.figure.colorbar(ax.get_images()[0], ax=ax, orientation='vertical', label='Logarithmic PSD')
    fig.tight_layout()

    return fig, ax

plot_spectrogram(emd_spectra_cond_avg_df)
plot_spectrogram(fft_spectra_cond_avg_df)

# %%
# plot just the 0.5 to 1.5 hz range
emd_spectra_cond_avg_df = pd.read_csv('Spectra/5-MeO-DMT_emd_E_spectrogram.csv', index_col=0)
fft_spectra_cond_avg_df = pd.read_csv('Spectra/5-MeO-DMT_fft_E_spectrogram.csv', index_col=0)

emd_spectra_cond_avg_df = np.log10(emd_spectra_cond_avg_df)
fft_spectra_cond_avg_df = np.log10(fft_spectra_cond_avg_df)

def plot_short_spectrogram(spectra_cond_avg_df):
    fig, ax = plt.subplots(1, 1, figsize=(13,2))
    cmap = sns.color_palette('gist_stern_r', as_cmap=True)
    import cmasher as cma
    cmap = cma.pride
    freq_centres = np.array(spectra_cond_avg_df.index)
    range = np.where((freq_centres <= 2.5))
    spectra_cond_avg_df = spectra_cond_avg_df.iloc[range]
    vmin = np.min(np.min(spectra_cond_avg_df))
    vmax = np.max(np.max(spectra_cond_avg_df))
    ax.imshow(spectra_cond_avg_df, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax.invert_yaxis()
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Freq (Hz)')
    ax.set_xticks([0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400], 
                  ['0', '2', '4', '6', '8', '10', '12', '14', '16', '18', '20'], rotation=0)
    gap = 3
    ax.set_yticks(range[0][::gap], np.round(freq_centres[range][::gap]-.2, 1), rotation=0)
    ax.tick_params(axis='x', which='major', pad=10)
    fig.tight_layout() 
    return fig, ax, vmin, vmax

def colbar_plot(vmin, vmax):
    fig, ax = plt.subplots(figsize=(0.1, 1.2))
    import matplotlib as mpl
    import cmasher as cma
    cmap = cma.pride
    norm = mpl.colors.Normalize(vmin=vmin , vmax=vmax)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')
    # make xticks and label90 degrees
    plt.xticks(rotation=90)
    cb1.set_label(r'LPSD', rotation=90)
    # rotate the labels
    plt.show()

fig, ax, vmin, vmax = plot_short_spectrogram(emd_spectra_cond_avg_df)
colbar_plot(vmin, vmax)
fig, ax, vmin, vmax = plot_short_spectrogram(fft_spectra_cond_avg_df)
colbar_plot(vmin, vmax)

# %%
emd_spectra_cond_avg_df.iloc[0:10]

# %%
emd_spectra_cond_avg_df = pd.read_csv('Spectra/5-MeO-DMT_emd_E_spectrogram.csv', index_col=0)
fft_spectra_cond_avg_df = pd.read_csv('Spectra/5-MeO-DMT_fft_E_spectrogram.csv', index_col=0)

full = five_full_epochs

def plot_spectra_heatmap(spectra_cond_avg_df, spect_range='full', vmin=None, vmax=None):
        import itertools
        from scipy import spatial 
        # convert to db
        spectra_cond_avg_df = 10*np.log10(spectra_cond_avg_df)# convert to db

        if spect_range != 'full':
                f1, f2 = spect_range
                # find the nearest frequency to f1 and f2 and their indexes
                freq_centres = np.array(spectra_cond_avg_df.index)
                f1_idx = np.argmin(np.abs(freq_centres - f1))
                f2_idx = np.argmin(np.abs(freq_centres - f2))
                # slice the dataframe
                spectra_cond_avg_df = spectra_cond_avg_df.iloc[f1_idx:f2_idx]

        num_freqs, num_epochs = spectra_cond_avg_df.shape
        distance_matrix = np.zeros((num_epochs, num_epochs))  
        for epoch1, epoch2 in itertools.combinations(full, 2):  
                spect_epoch1 = np.array(spectra_cond_avg_df[str(epoch1)])
                spect_epoch2 = np.array(spectra_cond_avg_df[str(epoch2)])       
                dist = spatial.distance.euclidean(spect_epoch1, spect_epoch2)   
                distance_matrix[epoch1, epoch2] = dist
                distance_matrix[epoch2, epoch1] = dist  
        
        # plot the distance matrix
        plt.figure(figsize=(6,5))       
        import cmasher as cma
        cmap = cma.ghostlight

        if vmin == None:
                vmin = np.percentile(distance_matrix, 1)
        if vmax == None:
                vmax = np.percentile(distance_matrix, 99)

        plt.imshow(distance_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        plt.gca().invert_yaxis()
        plt.xticks([0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400], ['0', '2', '4', '6', '8', '10', '12', '14', '16', '18', '20'], rotation=0)
        plt.yticks([0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400], ['0', '2', '4', '6', '8', '10', '12', '14', '16', '18', '20'], rotation=0)
        plt.xlabel('Time (minutes)')
        plt.ylabel('Time (minutes)')
        cbar = plt.colorbar()
        cbar.set_label('Euclidean Distance')    
        return()

plot_spectra_heatmap(emd_spectra_cond_avg_df, spect_range='full', vmin=0, vmax=100)
plot_spectra_heatmap(fft_spectra_cond_avg_df, spect_range='full', vmin=0, vmax=100)

plot_spectra_heatmap(emd_spectra_cond_avg_df, spect_range=[0.5, 4], vmin=None, vmax=None)
plot_spectra_heatmap(fft_spectra_cond_avg_df, spect_range=[0.5, 4], vmin=None, vmax=None)


# %%
# compute percentage change in power for the low frequency bands

ts = ['peak', 'Ppeak']

df = pd.read_csv(f'Spectra/{perps[0]}/{conditions[0]}/{perps[0]}_{conditions[0]}_E_emd_spectra_peak.csv', index_col=0)
peak_centres = np.array(df.index)
df = pd.read_csv(f'Spectra/{perps[0]}/{conditions[0]}/{perps[0]}_{conditions[0]}_E_emd_spectra_Ppeak.csv', index_col=0)
Ppeak_centres = np.array(df.index)

freq_edges, freq_centres = emd.spectra.define_hist_bins(0.5, 50, res, 'linear')
peak_edges = freq_edges[:2]
print(peak_edges)

for time in ts:
    if time == 'peak':
        centres = peak_centres
    else:
        centres = Ppeak_centres
    res = len(centres)

    rest_emd_spectra_df = pd.read_csv(f'Spectra/Rest_emd_E_spectra_{time}.csv', index_col=0)
    rest_emd_spectra_avg = rest_emd_spectra_df['avg'].values
    five_emd_spectra_df = pd.read_csv(f'Spectra/5-MeO-DMT_emd_E_spectra_{time}.csv', index_col=0)
    five_emd_spectra_avg = five_emd_spectra_df['avg'].values

    five_lowest = five_emd_spectra_avg[0]
    rest_lowest = rest_emd_spectra_avg[0]
    low_change = ((five_lowest - rest_lowest)/rest_lowest)*100
    print(f'{time} lowest frequency change: {low_change}')

# %%
def plot_spectra(rest_t_avg, rest_t_se, five_t_avg, five_t_se, res, y='linear'):

    freq_edges, freqs = emd.spectra.define_hist_bins(0.5, 50, res, 'linear')
    plt.figure(figsize=(5.5,5))
    plt.plot(freqs, rest_t_avg, label='Rest', color='green')
    plt.fill_between(freqs, rest_t_avg-rest_t_se, rest_t_avg+rest_t_se, color='green', alpha=0.3)
    plt.plot(freqs, five_t_avg, label='5-MeO-DMT', color='purple')
    plt.fill_between(freqs, five_t_avg-five_t_se, five_t_avg+five_t_se, color='purple', alpha=0.3)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(r'Power ($\mu V^2$)')
    plt.legend(loc='upper right', frameon=False)

    if y == 'log':
        plt.yscale('log')
        y_min = min(min(rest_t_avg-rest_t_se), min(five_t_avg-five_t_se))
    else:
        y_max = max(max(rest_t_avg+rest_t_se), max(five_t_avg+five_t_se))
        f = 20
        y_min = -y_max/f
        plt.ylim(y_min-(y_max/(f*2)), y_max+(y_max/(f*2)))

    for band in band_range_dict:
        plt.axvline(band_range_dict[band][0], color='grey', linestyle='--', linewidth=0.25, alpha=0.5)
        plt.axvline(band_range_dict[band][1], color='grey', linestyle='--', linewidth=0.25, alpha=0.5)
        pos = (band_range_dict[band][0]+band_range_dict[band][1])/2
        plt.text(pos, y_min, band_symbol_dict[band], color='grey', ha='center')

    plt.xscale('log')

    sns.despine()

    return()

# %%
ts = ['peak', 'Ppeak']

df = pd.read_csv(f'Spectra/{perps[0]}/{conditions[0]}/{perps[0]}_{conditions[0]}_E_emd_spectra_peak.csv', index_col=0)
peak_centres = np.array(df.index)
df = pd.read_csv(f'Spectra/{perps[0]}/{conditions[0]}/{perps[0]}_{conditions[0]}_E_emd_spectra_Ppeak.csv', index_col=0)
Ppeak_centres = np.array(df.index)

if not os.path.exists('Spectra/Rest_emd_E_spectra_peak.csv'):

    for time in ts:
        if time == 'peak':
            centres = peak_centres
        else:
            centres = Ppeak_centres
        for condition in conditions:
            emd_spectra_df = pd.DataFrame(columns=['avg', 'se'], index=centres)
            fft_spectra_df = pd.DataFrame(columns=['avg', 'se'], index=centres)
            for perp in perps:
                emd_spectra_perp_df = pd.read_csv(f'Spectra/{perp}/{condition}/{perp}_{condition}_E_emd_spectra_{time}.csv', index_col=0)
                fft_spectra_perp_df = pd.read_csv(f'Spectra/{perp}/{condition}/{perp}_{condition}_E_fft_spectra_{time}.csv', index_col=0)
                if perp == perps[0]:
                    emd_spectra_ext = emd_spectra_perp_df['avg'].values
                    fft_spectra_ext = fft_spectra_perp_df['avg'].values
                else:
                    emd_spectra_ext = np.vstack((emd_spectra_ext, emd_spectra_perp_df['avg'].values))
                    fft_spectra_ext = np.vstack((fft_spectra_ext, fft_spectra_perp_df['avg'].values))
            emd_spectra_df['avg'] = np.nanmean(emd_spectra_ext, axis=0)
            emd_spectra_df['se'] = np.nanstd(emd_spectra_ext, axis=0)/np.sqrt(len(perps))
            fft_spectra_df['avg'] = np.nanmean(fft_spectra_ext, axis=0)
            fft_spectra_df['se'] = np.nanstd(fft_spectra_ext, axis=0)/np.sqrt(len(perps))
            emd_spectra_df.to_csv(f'Spectra/{condition}_emd_E_spectra_{time}.csv')
            fft_spectra_df.to_csv(f'Spectra/{condition}_fft_E_spectra_{time}.csv')

for time in ts:
    if time == 'peak':
        centres = peak_centres
    else:
        centres = Ppeak_centres
    res = len(centres)

    rest_emd_spectra_df = pd.read_csv(f'Spectra/Rest_emd_E_spectra_{time}.csv', index_col=0)
    rest_emd_spectra_avg = rest_emd_spectra_df['avg']
    rest_emd_spectra_se = rest_emd_spectra_df['se']
    five_emd_spectra_df = pd.read_csv(f'Spectra/5-MeO-DMT_emd_E_spectra_{time}.csv', index_col=0)
    five_emd_spectra_avg = five_emd_spectra_df['avg']
    five_emd_spectra_se = five_emd_spectra_df['se']

    plot_spectra(rest_emd_spectra_avg, rest_emd_spectra_se, five_emd_spectra_avg, five_emd_spectra_se, res, y='log')

    rest_fft_spectra_df = pd.read_csv(f'Spectra/Rest_fft_E_spectra_{time}.csv', index_col=0)
    rest_fft_spectra_avg = rest_fft_spectra_df['avg']
    rest_fft_spectra_se = rest_fft_spectra_df['se']
    five_fft_spectra_df = pd.read_csv(f'Spectra/5-MeO-DMT_fft_E_spectra_{time}.csv', index_col=0)
    five_fft_spectra_avg = five_fft_spectra_df['avg']
    five_fft_spectra_se = five_fft_spectra_df['se']

    plot_spectra(rest_fft_spectra_avg, rest_fft_spectra_se, five_fft_spectra_avg, five_fft_spectra_se, res, y='log')


# %%
def oscillatory_peak(psd, freqs, sfreq, knee=True, freq_range=[0.5, 50],
                     band_range_dict={'alpha': [5,15]}):
    '''
    Computes the individual alpha peak frequency and power

    NB: FOOOF requires psd in linear space!

    '''
    import mne
    from fooof import FOOOF
    from fooof.bands import Bands
    from fooof.analysis import get_band_peak_fm

    fm = FOOOF(aperiodic_mode='knee')
    fm.fit(freqs, psd, freq_range)

    bands = Bands(band_range_dict)

    try:
        alphas = get_band_peak_fm(fm, bands.alpha, select_highest=True, attribute='gaussian_params')
        alpha_freq = alphas[0]
        alpha_pw = alphas[1]
    except Exception as e:
        alpha_freq = np.nan
        alpha_pw = np.nan

    try:
        ap = fm.get_params('aperiodic_params')
        if knee == True:
            ap_exp = ap[2]
        else:
            ap_exp = ap[1]
    except Exception as e:
        ap_exp = np.nan

    return alpha_pw, alpha_freq, ap_exp

fooof_df = pd.DataFrame(columns=['perp', 'condition', 'alpha_pw', 'alpha_freq', 'ap_exp'])

for perp in perps:
    for condition in conditions:
        emd_spectra_df = pd.read_csv(f'Spectra/{perp}/{condition}/{perp}_{condition}_E_emd_spectra_Ppeak.csv', index_col=0)
        emd_spectra = emd_spectra_df['avg'].values
        freqs = np.array(emd_spectra_df.index)
        alpha_pw, alpha_freq, ap_exp = oscillatory_peak(emd_spectra, freqs, sfreq, knee=True, freq_range=[0.5, 50])
        fooof_df = fooof_df.append({'perp': perp, 'condition': condition, 'alpha_pw': alpha_pw, 'alpha_freq': alpha_freq, 'ap_exp': ap_exp}, ignore_index=True)

fooof_stats_df = pd.DataFrame(columns=['measure', 't', 'p', 'cil', 'ciu', 'd', 'bf', 'dof', 'power'])
for measure in ['alpha_pw', 'alpha_freq', 'ap_exp']:
    t, p, cil, ciu, d, bf, dof, power = aug_t_test(fooof_df[fooof_df['condition'] == '5-MeO-DMT'][measure], fooof_df[fooof_df['condition'] == 'Rest'][measure])
    fooof_stats_df = fooof_stats_df.append({'measure': measure, 't': t, 'p': p, 'cil': cil, 'ciu': ciu, 'd': d, 'bf': bf, 'dof': dof, 'power': power}, ignore_index=True)
fooof_stats_df['p_bh'] = multi.multipletests(fooof_stats_df['p'], method='fdr_bh')[1]

# write stats for paper
for measure in ['alpha_pw', 'alpha_freq', 'ap_exp']:
    p = write_scientific(fooof_stats_df['p_bh'][fooof_stats_df['measure'] == measure].values[0])
    print(print(f'{measure}: $p_{{FDR}}=${p}.'))

# plot a boxplot of ap_exp
fig, ax = plt.subplots(figsize=(2,4))
sns.boxplot(x='condition', y='ap_exp', data=fooof_df, hue='condition', fill=False,
            palette={'Rest': 'green', '5-MeO-DMT': 'purple'}, ax=ax, width=0.5)
ax.set_ylabel('Exponent')
ax.set_xlabel('')
ax.set_xticks([0, 1], ['Rest', '5-MeO'])
sns.despine()

# plot a boxplot of alpha_pw 
fig, ax = plt.subplots(figsize=(2,4))
sns.boxplot(x='condition', y='alpha_pw', data=fooof_df, hue='condition', fill=False,
            palette={'Rest': 'green', '5-MeO-DMT': 'purple'}, ax=ax, width=0.5)
ax.set_ylabel(r'$\alpha$ Power')
ax.set_xlabel('')
ax.set_xticks([0, 1], ['Rest', '5-MeO'])
sns.despine()


# plot a boxplot of alpha_freq
fig, ax = plt.subplots(figsize=(2,4))
sns.boxplot(x='condition', y='alpha_freq', data=fooof_df, hue='condition', fill=False,
            palette={'Rest': 'green', '5-MeO-DMT': 'purple'}, ax=ax, width=0.5)
ax.set_ylabel(r'$\alpha$ Frequency')
ax.set_xlabel('')
ax.set_xticks([0, 1], ['Rest', '5-MeO'])
sns.despine()


# %%


# %%


# %%


# %%
if not os.path.exists(f'Spectra/emd_E_spectra_bands_{ts[1]}_stats.csv'):
    for time in ts:
        emd_band_stats_df = pd.DataFrame(columns=['band', 'electrode', 't', 'p', 'cil', 'ciu', 'd', 'bf', 'dof', 'power', 'p_bh'])
        fft_band_stats_df = pd.DataFrame(columns=['band', 'electrode', 't', 'p', 'cil', 'ciu', 'd', 'bf', 'dof', 'power', 'p_bh'])
        for band in band_range_dict.keys():
            for electrode in electrode_order:
                emd_rest_vals=[]; emd_five_vals=[]
                fft_rest_vals=[]; fft_five_vals=[]
                for perp in perps:
                    emd_rest_vals.append(pd.read_csv(f'Spectra/{perp}/Rest/{perp}_Rest_E_emd_spectra_bands_{time}.csv', index_col=0)[electrode][band])
                    emd_five_vals.append(pd.read_csv(f'Spectra/{perp}/5-MeO-DMT/{perp}_5-MeO-DMT_E_emd_spectra_bands_{time}.csv', index_col=0)[electrode][band])
                    fft_rest_vals.append(pd.read_csv(f'Spectra/{perp}/Rest/{perp}_Rest_E_fft_spectra_bands_{time}.csv', index_col=0)[electrode][band])
                    fft_five_vals.append(pd.read_csv(f'Spectra/{perp}/5-MeO-DMT/{perp}_5-MeO-DMT_E_fft_spectra_bands_{time}.csv', index_col=0)[electrode][band])
                T, p, cil, ciu, d, bf, dof, power = aug_t_test(emd_five_vals, emd_rest_vals)
                emd_band_stats_df = emd_band_stats_df.append({'band': band, 'electrode': electrode, 't': T, 'p': p, 'cil': cil, 'ciu': ciu, 'd': d, 'bf': bf, 'dof': dof, 'power': power}, ignore_index=True)
                T, p, cil, ciu, d, bf, dof, power = aug_t_test(fft_five_vals, fft_rest_vals)
                fft_band_stats_df = fft_band_stats_df.append({'band': band, 'electrode': electrode, 't': T, 'p': p, 'cil': cil, 'ciu': ciu, 'd': d, 'bf': bf, 'dof': dof, 'power': power}, ignore_index=True)
        emd_band_stats_df['p_bh'] = multi.multipletests(emd_band_stats_df['p'], method='fdr_bh')[1]
        emd_band_stats_df.to_csv(f'Spectra/emd_E_spectra_bands_{time}_stats.csv')
        fft_band_stats_df['p_bh'] = multi.multipletests(fft_band_stats_df['p'], method='fdr_bh')[1]
        fft_band_stats_df.to_csv(f'Spectra/fft_E_spectra_bands_{time}_stats.csv')

# %%
# whats the maximum T and other respective stats
emd_band_stats_df = pd.read_csv('Spectra/emd_E_spectra_bands_Ppeak_stats.csv', index_col=0)
max_idx = emd_band_stats_df[emd_band_stats_df['band'] == 'slow']['t'].idxmax()
print(emd_band_stats_df.iloc[max_idx])

# %%
# whats the maximum T and other respective stats
emd_band_stats_df = pd.read_csv('Spectra/emd_E_spectra_bands_Ppeak_stats.csv', index_col=0)
max_idx = emd_band_stats_df[emd_band_stats_df['band'] == 'gamma']['t'].idxmax()
print(emd_band_stats_df.iloc[max_idx])

# %%
emd_band_stats_df = pd.read_csv('Spectra/emd_E_spectra_bands_Ppeak_stats.csv', index_col=0)
max_idx = emd_band_stats_df[emd_band_stats_df['band'] == 'alpha']['t'].idxmax()
print(emd_band_stats_df.iloc[max_idx])

# %%
emd_band_stats_df = pd.read_csv('Spectra/emd_E_spectra_bands_Ppeak_stats.csv', index_col=0)
max_idx = emd_band_stats_df[emd_band_stats_df['band'] == 'alpha']['t'].idxmin()
print(emd_band_stats_df.iloc[max_idx])

# %%
def plot_topoplot_bands(method, time='peak', ext='E', split=True, vmin=None, vmax=None):
    if vmin == None:
        vmin = -5
    if vmax == None:
        vmax = 5
    
    if split == True:
        fig, axs = plt.subplots(2, 3, figsize=(7, 5))
        es = 3
        # title loc is at top
        title_loc = 1
    else:
        fig, axs = plt.subplots(1, 6, figsize=(12, 5))
        es = 2
        # title loc is at bottom
        title_loc = -0.2
    fs = 14
    band_stats = pd.read_csv('Spectra/'+method+'_E_spectra_bands_'+time+'_stats.csv', index_col=0)

    for i, band in enumerate(band_range_dict.keys()):

        if split == True:
            if i < 3:
                ax = axs[0, i]
            else:
                ax = axs[1, i-3]
        else:
            ax = axs[i]

        tvals = []
        pvals = []

        for electrode in ch_names:
            tvals.append(band_stats[band_stats['band']==band][band_stats['electrode']==electrode]['t'].values[0])
            pvals.append(band_stats[band_stats['band']==band][band_stats['electrode']==electrode]['p_bh'].values[0])

        # if a p-value is less than 0.05 then set it to 1 otherwise 0
        for i in range(len(pvals)):
            if pvals[i] < 0.05:
                pvals[i] = 1
            else:
                pvals[i] = 0
        pvals = np.array(pvals)

        mne.viz.plot_topomap(tvals, raw.info, cmap='PRGn_r', vlim=(vmin, vmax),
            sensors=False, outlines='head', contours=6, sphere=0.1, ch_type='eeg', res=64, image_interp='linear', axes=ax, show=False,
            mask=pvals, mask_params=dict(marker='o', markerfacecolor='white', markeredgecolor='black', linewidth=0, markersize=es))

        # set titles
        if band == 'slow':
            if ext == 'E':
                ax.set_title(r'$Slow$ 0.5-1.5 Hz', fontsize=fs, y=title_loc)
            if ext == 'F':
                ax.set_title(r'$Slow$ 0.1-1.5 Hz', fontsize=fs, y=title_loc)
        elif band == 'delta':
            ax.set_title(r'$\delta$ 1.5-4 Hz', fontsize=fs, y=title_loc)
        elif band == 'theta':
            ax.set_title(r'$\theta$ 4-8 Hz', fontsize=fs, y=title_loc)
        elif band == 'alpha':
            ax.set_title(r'$\alpha$ 8-12 Hz', fontsize=fs, y=title_loc)
        elif band == 'beta':
            ax.set_title(r'$\beta$ 12-30 Hz', fontsize=fs, y=title_loc)
        elif band == 'gamma':
            ax.set_title(r'$\gamma$ 30-50 Hz', fontsize=fs, y=title_loc)


    fig.tight_layout()

    return()

plot_topoplot_bands('emd', 'peak', 'E')
plot_topoplot_bands('emd', 'peak', 'E', split=False)
plot_topoplot_bands('fft', 'peak', 'E')
plot_topoplot_bands('fft', 'peak', 'E', split=False)
plot_topoplot_bands('emd', 'Ppeak', 'E', split=True, vmin=-4, vmax=4)
plot_topoplot_bands('emd', 'Ppeak', 'E', split=False, vmin=-4, vmax=4)
plot_topoplot_bands('fft', 'Ppeak', 'E', split=True, vmin=-4, vmax=4)
plot_topoplot_bands('fft', 'Ppeak', 'E', split=False, vmin=-4, vmax=4)

# %%
fig, ax = plt.subplots(figsize=(0.15, 2))
import matplotlib as mpl
cmap = mpl.cm.PRGn_r
norm = mpl.colors.Normalize(vmin=-4, vmax=4)
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')
plt.xticks(rotation=90)
cb1.set_ticks([-4, 0, 4])
cb1.set_ticklabels(['-4', '0', '4'], fontsize=18)
cb1.set_label(r'$T$', rotation=0, fontsize=18)
plt.show()

# %%
ts = ['peak', 'Ppeak']
for time in ts:
    from scipy.stats import ttest_rel
    from mne.stats import permutation_cluster_test, permutation_cluster_1samp_test

    fam = np.linspace(0.5, 4, 50)
    fcarrier = np.linspace(8, 50, 50)

    carrier_res = 50; am_res = 50
    rest_holo = np.array(np.nan * np.zeros((carrier_res, am_res, len(perps))))
    five_holo = np.array(np.nan * np.zeros((carrier_res, am_res, len(perps))))

    for i, condition in enumerate(conditions):
        for j, perp in enumerate(perps):
            # Holo-spectrum is in form (carrierfreq x amfreq)
            holo = np.load(f'Spectra/{perp}/{condition}/{perp}_{condition}_E_{time}_holo.npy')
            if i == 0:
                rest_holo[:, :, j] = holo
            else:
                five_holo[:, :, j] = holo

    # mne requires a 3d array of shape (perps, carrier, am)
    rest_holo_ = np.transpose(rest_holo, (2, 0, 1))
    five_holo_ = np.transpose(five_holo, (2, 0, 1))

    n_perm = 1000
    t_obs, clusters, cluster_pv, H0 = permutation_cluster_test([five_holo_, rest_holo_], 
                                                               threshold=None, tail=0,
                                                               n_permutations=n_perm, adjacency=None, 
                                                               n_jobs=1, seed=50)

    # print clusters
    for c, p_val in zip(clusters, cluster_pv):
        if p_val <= 0.05:
            print(len(c[0]), p_val)

    # create new stats image with only significant clusters
    pthresh = 0.05
    t_obs_plot = np.zeros(t_obs.shape)
    for c, p_val in zip(clusters, cluster_pv):
        if p_val <= pthresh:
            t_obs_plot[c] = t_obs[c]

    vmax = np.percentile(t_obs, 99)
    vmax_ = np.max(np.abs(t_obs))

    # plot cluster results
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.75))
    # plt.pcolormesh(fam, fcarrier, t_obs, cmap='gray_r', shading='auto', vmin=0, vmax=vmax_)
    plt.pcolormesh(fam, fcarrier, t_obs_plot,
                    cmap='inferno', shading='auto', vmin=0, vmax=vmax)
    plt.ylabel('Carrier Freq (Hz)')
    plt.xlabel('AM Freq (Hz)')
    plt.colorbar(label=r'$\Delta$ AM ($F$)')

    # just inspect with T to confirm direction of effect
    plt.figure(figsize=(6, 4.75))
    T_res = np.nan*np.zeros((carrier_res, am_res))
    for i in range(carrier_res):
        for j in range(am_res):
            T_res[i, j] = ttest_rel(five_holo[i, j, :], rest_holo[i, j, :])[0]
    vmx = np.percentile(np.abs(T_res), 99)
    plt.pcolormesh(fam, fcarrier, T_res, cmap='PRGn_r', shading='auto', vmin=-vmx, vmax=vmx)  
    plt.ylabel('Carrier Freq (Hz)')
    plt.xlabel('AM Freq (Hz)')
    plt.colorbar(label=r'$\Delta$ AM ($T$)')
    

# %%
#######
# F ###
#######

# %%
ts = ['peak', 'Ppeak']

df = pd.read_csv(f'Spectra/{perps[0]}/{conditions[0]}/{perps[0]}_{conditions[0]}_F_emd_spectra_peak.csv', index_col=0)
peak_centres = np.array(df.index)
df = pd.read_csv(f'Spectra/{perps[0]}/{conditions[0]}/{perps[0]}_{conditions[0]}_F_emd_spectra_Ppeak.csv', index_col=0)
Ppeak_centres = np.array(df.index)

if not os.path.exists('Spectra/Rest_emd_F_spectra_peak.csv'):

    for time in ts:
        if time == 'peak':
            centres = peak_centres
        else:
            centres = Ppeak_centres
        for condition in conditions:
            emd_spectra_df = pd.DataFrame(columns=['avg', 'se'], index=centres)
            fft_spectra_df = pd.DataFrame(columns=['avg', 'se'], index=centres)
            for perp in peak_perps:
                emd_spectra_perp_df = pd.read_csv(f'Spectra/{perp}/{condition}/{perp}_{condition}_F_emd_spectra_{time}.csv', index_col=0)
                fft_spectra_perp_df = pd.read_csv(f'Spectra/{perp}/{condition}/{perp}_{condition}_F_fft_spectra_{time}.csv', index_col=0)
                if perp == perps[0]:
                    emd_spectra_ext = emd_spectra_perp_df['avg'].values
                    fft_spectra_ext = fft_spectra_perp_df['avg'].values
                else:
                    emd_spectra_ext = np.vstack((emd_spectra_ext, emd_spectra_perp_df['avg'].values))
                    fft_spectra_ext = np.vstack((fft_spectra_ext, fft_spectra_perp_df['avg'].values))
            emd_spectra_df['avg'] = np.nanmean(emd_spectra_ext, axis=0)
            emd_spectra_df['se'] = np.nanstd(emd_spectra_ext, axis=0)/np.sqrt(len(peak_perps))
            fft_spectra_df['avg'] = np.nanmean(fft_spectra_ext, axis=0)
            fft_spectra_df['se'] = np.nanstd(fft_spectra_ext, axis=0)/np.sqrt(len(peak_perps))
            emd_spectra_df.to_csv(f'Spectra/{condition}_emd_F_spectra_{time}.csv')
            fft_spectra_df.to_csv(f'Spectra/{condition}_fft_F_spectra_{time}.csv')

for time in ts:
    if time == 'peak':
        centres = peak_centres
    else:
        centres = Ppeak_centres
    res = len(centres)

    rest_emd_spectra_df = pd.read_csv(f'Spectra/Rest_emd_F_spectra_{time}.csv', index_col=0)
    rest_emd_spectra_avg = rest_emd_spectra_df['avg']
    rest_emd_spectra_se = rest_emd_spectra_df['se']
    five_emd_spectra_df = pd.read_csv(f'Spectra/5-MeO-DMT_emd_F_spectra_{time}.csv', index_col=0)
    five_emd_spectra_avg = five_emd_spectra_df['avg']
    five_emd_spectra_se = five_emd_spectra_df['se']

    plot_spectra(rest_emd_spectra_avg, rest_emd_spectra_se, five_emd_spectra_avg, five_emd_spectra_se, res, y='log')

    rest_fft_spectra_df = pd.read_csv(f'Spectra/Rest_fft_F_spectra_{time}.csv', index_col=0)
    rest_fft_spectra_avg = rest_fft_spectra_df['avg']
    rest_fft_spectra_se = rest_fft_spectra_df['se']
    five_fft_spectra_df = pd.read_csv(f'Spectra/5-MeO-DMT_fft_F_spectra_{time}.csv', index_col=0)
    five_fft_spectra_avg = five_fft_spectra_df['avg']
    five_fft_spectra_se = five_fft_spectra_df['se']

    plot_spectra(rest_fft_spectra_avg, rest_fft_spectra_se, five_fft_spectra_avg, five_fft_spectra_se, res, y='log')

# %%
# want to test for significant differences in the bands between the two conditions at each electrode

if not os.path.exists('Spectra/gamma_emd_F_spectra_peak_stats.csv'):
    for time in ts:
        emd_band_stats_df = pd.DataFrame(columns=['band', 'electrode', 't', 'p', 'cil', 'ciu', 'd', 'bf', 'dof', 'power', 'p_bh'])
        fft_band_stats_df = pd.DataFrame(columns=['band', 'electrode', 't', 'p', 'cil', 'ciu', 'd', 'bf', 'dof', 'power', 'p_bh'])
        for band in band_range_dict.keys():
            for region, electrode in enumerate(electrode_order):
                emd_rest_vals=[]; emd_five_vals=[]
                fft_rest_vals=[]; fft_five_vals=[]
                for perp in peak_perps:
                    emd_rest_vals.append(pd.read_csv(f'Spectra/{perp}/Rest/{perp}_Rest_F_emd_spectra_bands_{time}.csv', index_col=0)[str(region)][band])
                    emd_five_vals.append(pd.read_csv(f'Spectra/{perp}/5-MeO-DMT/{perp}_5-MeO-DMT_F_emd_spectra_bands_{time}.csv', index_col=0)[str(region)][band])
                    fft_rest_vals.append(pd.read_csv(f'Spectra/{perp}/Rest/{perp}_Rest_F_fft_spectra_bands_{time}.csv', index_col=0)[str(region)][band])
                    fft_five_vals.append(pd.read_csv(f'Spectra/{perp}/5-MeO-DMT/{perp}_5-MeO-DMT_F_fft_spectra_bands_{time}.csv', index_col=0)[str(region)][band])
                T, p, cil, ciu, d, bf, dof, power = aug_t_test(emd_five_vals, emd_rest_vals)
                emd_band_stats_df = emd_band_stats_df.append({'band': band, 'electrode': electrode, 't': T, 'p': p, 'cil': cil, 'ciu': ciu, 'd': d, 'bf': bf, 'dof': dof, 'power': power}, ignore_index=True)
                T, p, cil, ciu, d, bf, dof, power = aug_t_test(fft_five_vals, fft_rest_vals)
                fft_band_stats_df = fft_band_stats_df.append({'band': band, 'electrode': electrode, 't': T, 'p': p, 'cil': cil, 'ciu': ciu, 'd': d, 'bf': bf, 'dof': dof, 'power': power}, ignore_index=True)
        emd_band_stats_df['p_bh'] = multi.multipletests(emd_band_stats_df['p'], method='fdr_bh')[1]
        emd_band_stats_df.to_csv(f'Spectra/emd_F_spectra_bands_{time}_stats.csv')
        fft_band_stats_df['p_bh'] = multi.multipletests(fft_band_stats_df['p'], method='fdr_bh')[1]
        fft_band_stats_df.to_csv(f'Spectra/fft_F_spectra_bands_{time}_stats.csv')

# %%
plot_topoplot_bands('emd', 'peak', 'F', split=False)
plot_topoplot_bands('fft', 'peak', 'F', split=False)
plot_topoplot_bands('emd', 'Ppeak', 'F', split=False)
plot_topoplot_bands('fft', 'Ppeak', 'F', split=False)

# %%
for time in ts:
    from mne.stats import permutation_cluster_test, permutation_cluster_1samp_test

    fam = np.linspace(0.1, 4, 50)
    fcarrier = np.linspace(8, 50, 50)

    carrier_res = 50; am_res = 50
    rest_holo = np.array(np.nan * np.zeros((carrier_res, am_res, len(peak_perps))))
    five_holo = np.array(np.nan * np.zeros((carrier_res, am_res, len(peak_perps))))

    for i, condition in enumerate(conditions):
        for j, perp in enumerate(peak_perps):
            # Holo-spectrum is in form (carrierfreq x amfreq)
            holo = np.load(f'Spectra/{perp}/{condition}/{perp}_{condition}_F_{time}_holo.npy')
            if i == 0:
                rest_holo[:, :, j] = np.mean(holo, axis=2)
            else:
                five_holo[:, :, j] = np.mean(holo, axis=2)

    # plot T results
    plt.figure(figsize=(6, 5))
    T_res = np.nan*np.zeros((carrier_res, am_res))
    for i in range(carrier_res):
        for j in range(am_res):
            T_res[i, j] = scipy.stats.ttest_rel(five_holo[i, j, :], rest_holo[i, j, :])[0]
    vmx = np.percentile(np.abs(T_res), 99)
    plt.pcolormesh(fam, fcarrier, T_res, cmap='PRGn_r', shading='auto', vmin=-vmx, vmax=vmx)  
    plt.ylabel('Carrier Freq (Hz)')
    plt.xlabel('AM Freq (Hz)')
    plt.colorbar(label=r'$\Delta$ PAC')

    # mne requires a 3d array of shape (perps, carrier, am)
    rest_holo_ = np.transpose(rest_holo, (2, 0, 1))
    five_holo_ = np.transpose(five_holo, (2, 0, 1))

    n_perm = 1000
    t_obs, clusters, cluster_pv, H0 = permutation_cluster_test([five_holo_, rest_holo_], 
                                                               threshold=None, tail=0,
                                                               n_permutations=n_perm, adjacency=None, n_jobs=1)

    # print clusters
    for c, p_val in zip(clusters, cluster_pv):
        if p_val <= 0.05:
            print(len(c[0]), p_val)

    # create new stats image with only significant clusters
    pthresh = 0.05
    t_obs_plot = np.nan*np.zeros(t_obs.shape)
    for c, p_val in zip(clusters, cluster_pv):
        if p_val <= pthresh:
            t_obs_plot[c] = t_obs[c]

    vmax = np.percentile(t_obs, 99)
    vmax_ = np.max(np.abs(t_obs))

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plt.pcolormesh(fam, fcarrier, t_obs,
                    cmap='gray_r', shading='auto', vmin=0, vmax=vmax_)
    plt.pcolormesh(fam, fcarrier, t_obs_plot,
                    cmap='pink_r', shading='auto', vmin=0, vmax=vmax)
    plt.ylabel('Carrier Freq (Hz)')
    plt.xlabel('AM Freq (Hz)')
    plt.colorbar(label=r'$\Delta$ AM ($F$)')


