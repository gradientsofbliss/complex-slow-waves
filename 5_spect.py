import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import mne; mne.set_log_level('ERROR')
import emd

perps = ['PCJ09', 'PFM82', 'PDH47', 'PLM37', 'PJS04', 'PVU29', 'PUA35', 'PTS72', 'PXH23', 'PQS29', 'P2M63', 'P5P11', 'PGH44', 'P4O85', 'P3N74', 'PBA32', 'PDL71', 'P7R50', 'PFQ62']

def fft_spectra(timeseries, sfreq=500,
                fmin=0.5, fmax=50, 
                form='linear'):
    """
    Compute multi-taper FFT power spectra

    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5343535/

    Parameters
    ----------
    timeseries : array
        The timeseries to compute the alpha power of
    sfreq : int
        The sampling frequency in Hz
    form : str
        The form of the output, either 'linear', 'log', or 'dB'

    Returns
    -------
    psd : array
        The power spectral density
    freqs : array
        The frequencies for the power spectral density
    """

    import mne

    timeseries = np.array(timeseries) # ensure is an array

    # normed by sfreq and length
    psd, freqs = mne.time_frequency.psd_array_multitaper(timeseries, sfreq=sfreq, fmin=fmin, fmax=fmax,
                                                         normalization='full', verbose=0, n_jobs=-1)

    if form == 'linear':
        psd = psd
    if form == 'log':
        psd = np.log10(psd)
    elif form == 'dB':
        psd = 10 * np.log10(psd)

    return psd, freqs

def emd_imfs(timeseries, sfreq=500,
             max_imfs=6,
             save=False, **kwargs):
    '''
    Get intrinsic mode functions

    Parameters
    ----------
    timeseries : array
        timeseries to decompose
    sfreq : int
        sampling frequency in Hz
    max_imfs : int
        maximum number of IMFs to compute
    save : bool
        whether to save the instantaneous properties

    Returns
    -------
    imfs : array
        
    '''

    import emd

    # ensure is an array
    timeseries = np.array(timeseries)

    imfs = emd.sift.iterated_mask_sift(timeseries, max_imfs=max_imfs)
    
    # chuck faulty drift imfs
    if np.isnan(imfs[0]).any():
        bad_index = np.where(np.isnan(imfs[0]))[0]
        imfs = imfs[:,:bad_index[0]]

    if save:
        IP, IF, IA = emd.spectra.frequency_transform(imfs, sfreq, 'nht')
        if not os.path.exists(kwargs['path']):
            os.makedirs(kwargs['path'])
        np.save(kwargs['path']+'imfs.npy', imfs)
        np.save(kwargs['path']+'IP.npy', IP)
        np.save(kwargs['path']+'IF.npy', IF)
        np.save(kwargs['path']+'IA.npy', IA)

    return imfs

def emd_spectra(imfs, IA, IF, mode='imfs', 
                sfreq=500, res=149, fmin=0.5, fmax=50, 
                form='linear'):
    '''
    Compute Hilbert-Huang Transform EMD power spectra

    https://royalsocietypublishing.org/doi/10.1098/rspa.1998.0193

    Parameters
    ----------
    timeseries : array
        The timeseries to compute the alpha power of
    sfreq : int
        The sampling frequency in Hz
    form : str
        The form of the output, either 'linear', 'log', or 'dB'

    Returns
    -------
    psd : array
        The power spectral density
    freqs : array
        The frequencies for the power spectral density
    '''

    import emd

    if mode == 'imfs':
        IP, IF, IA = emd.spectra.frequency_transform(imfs, sfreq, 'nht')

    freq_edges, freq_centres = emd.spectra.define_hist_bins(fmin, fmax, res, 'linear')
    f, hht = emd.spectra.hilberthuang(IF, IA, freq_edges, scaling='density')

    if form == 'linear':
        psd = hht
    if form == 'log':
        psd = np.log10(hht)
    elif form == 'dB':
        psd = 10 * np.log10(hht)

    return psd, freq_centres


def get_bands(psd, freqs, band_range_dict=None, out='array'):
    '''
    Return the avg power in frequency bands 

    Parameters
    ----------
    psd : array
        The power spectral density
    freqs : array
        The frequencies for the power spectral density
    band_range_dict : dict
        The frequency range for the band

    Returns
    -------
    bands : array or dict
        The average power in frequency bands
    '''

    if band_range_dict is None:
        band_range_dict = {'slow': [0.5,1.5], 'delta': [1.5,4], 'theta': [4,8], 'alpha': [8,12], 'beta': [12,30], 'gamma': [30,50]}

    bands = {}
    for band in band_range_dict:
        bands[band] = np.mean(psd[(freqs >= band_range_dict[band][0]) & (freqs < band_range_dict[band][1])])

    if out == 'array':
        bands = np.array(list(bands.values()))
    
    return bands

def fix_edge_case(spectra):
    if np.isnan(spectra).any():
        for i in range(len(spectra)):
            if np.isnan(spectra[i]):
                if i < 3:
                    spectra[i] = spectra[np.where(~np.isnan(spectra))[0][0]]
                elif i > len(spectra)-3:
                    spectra[i] = spectra[np.where(~np.isnan(spectra))[0][-1]]
                else:
                    if np.isnan(spectra[i-1]) and ~np.isnan(spectra[i+1]):
                        spectra[i] = np.nanmean([spectra[i-2], spectra[i+1]])
                    elif ~np.isnan(spectra[i-1]) and np.isnan(spectra[i+1]):
                        spectra[i] = np.nanmean([spectra[i-1], spectra[i+2]])
                    else:
                        spectra[i] = np.nanmean([spectra[i-1], spectra[i+1]])
    return spectra

def get_holo(imfs, sfreq,
                 am_lf=.1, am_hf=4, am_res=50, 
                 carrier_lf=8, carrier_hf=50, carrier_res=50):
    
    IP, IF, IA = emd.spectra.frequency_transform(imfs, sfreq, 'nht')
    
    masks = np.array([25/2**ii for ii in range(12)])/sfreq
    config = emd.sift.get_config('mask_sift')
    config['mask_amp_mode'] = 'ratio_sig'
    config['mask_amp'] = 2
    config['max_imfs'] = 5
    config['imf_opts/sd_thresh'] = 0.05
    config['envelope_opts/interp_method'] = 'mono_pchip'
    imfs2 = emd.sift.mask_sift_second_layer(IA, masks, sift_args=config)
    IP2, IF2, IA2 = emd.spectra.frequency_transform(imfs2, sfreq, 'nht')

    am_hist = (am_lf, am_hf, am_res, 'linear')
    carrier_hist = (carrier_lf, carrier_hf, carrier_res, 'linear')

    # Holo-spectrum is in form (carrierfreq x amfreq)
    fcarrier, fam, holo = emd.spectra.holospectrum(IF, IF2, IA2, carrier_hist, am_hist)

    return fcarrier, fam, holo

holosp = True

# make a text file to store errors
if not os.path.exists('Spectra'):
    os.makedirs('Spectra')
with open('Spectra/Errors.txt', 'w') as f:
    f.write('')


def parallel_process(perp):

    conditions = ['Rest', '5-MeO-DMT']
    sfreq = 500
    n_regions = 64
    
    rest_full_epochs = np.arange(0,140)
    five_full_epochs = np.arange(0,400)
    
    good_epoch_dict = np.load('good_epoch_dict.npy', allow_pickle=True).item()
    good_perp_epoch_dict = np.load('good_perp_epoch_dict.npy', allow_pickle=True).item()
    
    electrode_list = np.load('electrode_order.npy', allow_pickle=True)

    for perp in perp:

        for condition in conditions:

            if not os.path.exists('Spectra/'+perp+'/'+condition):
                os.makedirs('Spectra/'+perp+'/'+condition)

            if not os.path.exists('Spectra/'+perp+'/'+condition+'/'+perp+'_'+condition+'_E_emd_spectra_bands_Ppeak.csv'):

                data = pd.read_csv('Data/Clean/Seg/'+perp+'/'+condition+'/'+perp+'_'+condition+'_seg.csv', index_col=0)
                # epochs = data['epoch'].unique()

                if condition == 'Rest':
                    full = rest_full_epochs
                else:
                    full = five_full_epochs
                good_epochs = good_epoch_dict[perp, condition]

                res = 149
                fmin = 0.5; fmax = 50
                band_range_dict = {'slow': [0.5,1.5], 'delta': [1.5,4], 'theta': [4,8], 'alpha': [8,12], 'beta': [12,30], 'gamma': [30,50]}

                freq_edges, freq_centres = emd.spectra.define_hist_bins(fmin, fmax, res, 'linear')

                am_res = 50; carrier_res = 50
                am_lf = .5; am_hf = 4
                carrier_lf = 8; carrier_hf = 50

                # mega arrays
                emd_spectra_array = np.nan * np.zeros((len(electrode_list), len(freq_centres), len(full)))
                fft_spectra_array = np.nan * np.zeros((len(electrode_list), len(freq_centres), len(full)))
                emd_spectra_bands_array = np.nan * np.zeros((len(electrode_list), len(band_range_dict), len(full)))
                fft_spectra_bands_array = np.nan * np.zeros((len(electrode_list), len(band_range_dict), len(full)))

                if holosp:
                    holospectrum_array = np.nan * np.zeros((carrier_res, am_res, len(electrode_list), len(full)))

                for epoch in good_epochs:

                    data_epoch = data[data['epoch'] == epoch]
                    data_epoch = data_epoch[electrode_list]

                    emd_spectra_df = pd.DataFrame(columns=electrode_list, index=freq_centres)
                    fft_spectra_df = pd.DataFrame(columns=electrode_list, index=freq_centres)

                    for electrode in electrode_list:
                        data_electrode = np.array(data_epoch[electrode])

                        imfs = emd_imfs(data_electrode, sfreq=sfreq)
                        emd_spect, freq_centres = emd_spectra(imfs, None, None, mode='imfs', sfreq=sfreq, res=res, fmin=fmin, fmax=fmax, form='linear')

                        fft_spect, freqs = fft_spectra(data_electrode, sfreq=sfreq, fmin=fmin, fmax=fmax, form='linear')

                        if np.isnan(emd_spect).any():
                            emd_spect = fix_edge_case(emd_spect)

                        if np.isnan(emd_spect).any():
                            with open('Spectra/Errors.txt', 'a') as f:
                                f.write(perp+' '+condition+' '+str(epoch)+' '+electrode+' EMD nans\n')
                        elif len(emd_spect) < len(freq_centres):
                            with open('Spectra/Errors.txt', 'a') as f:
                                f.write(perp+' '+condition+' '+str(epoch)+' '+electrode+' EMD short\n')
                        elif len(emd_spect) > len(freq_centres):
                            with open('Spectra/Errors.txt', 'a') as f:
                                f.write(perp+' '+condition+' '+str(epoch)+' '+electrode+' EMD long\n')
                        else:
                            emd_spectra_df[electrode] = emd_spect
                        if np.isnan(fft_spect).any():
                            with open('Spectra/Errors.txt', 'a') as f:
                                f.write(perp+' '+condition+' '+str(epoch)+' '+electrode+' FFT nans\n')
                        elif len(fft_spect) < len(freq_centres):
                            with open('Spectra/Errors.txt', 'a') as f:
                                f.write(perp+' '+condition+' '+str(epoch)+' '+electrode+' FFT short\n')
                        elif len(fft_spect) > len(freq_centres):
                            with open('Spectra/Errors.txt', 'a') as f:
                                f.write(perp+' '+condition+' '+str(epoch)+' '+electrode+' FFT long\n')
                        else:
                            fft_spectra_df[electrode] = fft_spect

                        electrode_index = np.where(electrode_list == electrode)[0][0]

                        emd_spectra_array[electrode_index, :, epoch] = emd_spect
                        fft_spectra_array[electrode_index, :, epoch] = fft_spect
                        emd_spectra_bands_array[electrode_index, :, epoch] = get_bands(emd_spect, freq_centres, band_range_dict)
                        fft_spectra_bands_array[electrode_index, :, epoch] = get_bands(fft_spect, freqs, band_range_dict)

                        if holosp:
                            # try but if it fails then skip

                            try:
                                fcarrier, fam, holo = get_holo(imfs, sfreq,
                                                            am_lf=am_lf, am_hf=am_hf, am_res=am_res, 
                                                            carrier_lf=carrier_lf, carrier_hf=carrier_hf, carrier_res=carrier_res)
                                holospectrum_array[:, :, electrode_index, epoch] = holo
                            except:
                                with open('Spectra/Errors.txt', 'a') as f:
                                    f.write(perp+' '+condition+' '+str(epoch)+' '+electrode+' holosp fail\n')
                                holo = np.nan * np.zeros((carrier_res, am_res))
                                holospectrum_array[:, :, electrode_index, epoch] = holo

                # average across electrodes so we can make a freq x epoch dataframe
                emd_spectra_avg_ = np.mean(emd_spectra_array, axis=0)
                fft_spectra_avg_ = np.mean(fft_spectra_array, axis=0)
                emd_spectra_avg_df = pd.DataFrame(index=freq_centres, columns=np.arange(len(full)), data=emd_spectra_avg_)
                fft_spectra_avg_df = pd.DataFrame(index=freq_centres, columns=np.arange(len(full)), data=fft_spectra_avg_)
                emd_spectra_avg_df.to_csv('Spectra/'+perp+'/'+condition+'/'+perp+'_'+condition+'_E_emd_spectrogram.csv')
                fft_spectra_avg_df.to_csv('Spectra/'+perp+'/'+condition+'/'+perp+'_'+condition+'_E_fft_spectrogram.csv')

                if holosp:
                    holospectrum_array_avg = np.nanmean(holospectrum_array, axis=2)

                margins = [[10,80], [30,50]]
                for margin in margins:
                    if margin == [10,80]:
                        name='peak'
                    else:
                        name='Ppeak'

                    #full (produce a single average spectrum)
                    # average over electrodes
                    emd_spectra_avg_ = np.mean(emd_spectra_array, axis=0)
                    fft_spectra_avg_ = np.mean(fft_spectra_array, axis=0)
                    # average over epochs
                    emd_spectra_avg = np.nanmean(emd_spectra_avg_[:, margin[0]:margin[1]], axis=1)
                    fft_spectra_avg = np.nanmean(fft_spectra_avg_[:, margin[0]:margin[1]], axis=1)
                    #
                    emd_spectra_avg_df = pd.DataFrame(index=freq_centres, columns=['avg'], data=emd_spectra_avg)
                    fft_spectra_avg_df = pd.DataFrame(index=freq_centres, columns=['avg'], data=fft_spectra_avg)
                    emd_spectra_avg_df.to_csv('Spectra/'+perp+'/'+condition+'/'+perp+'_'+condition+'_E_emd_spectra_'+str(name)+'.csv')
                    fft_spectra_avg_df.to_csv('Spectra/'+perp+'/'+condition+'/'+perp+'_'+condition+'_E_fft_spectra_'+str(name)+'.csv')

                    #bands (produce bands for each electrode)
                    # average over epochs
                    emd_spectra_bands_avg = np.nanmean(emd_spectra_bands_array[:, :, margin[0]:margin[1]], axis=2)
                    fft_spectra_bands_avg = np.nanmean(fft_spectra_bands_array[:, :, margin[0]:margin[1]], axis=2)
                    #
                    emd_spectra_bands_df = pd.DataFrame(index=band_range_dict.keys(), columns=electrode_list, data=emd_spectra_bands_avg.T)
                    fft_spectra_bands_df = pd.DataFrame(index=band_range_dict.keys(), columns=electrode_list, data=fft_spectra_bands_avg.T)
                    emd_spectra_bands_df.to_csv('Spectra/'+perp+'/'+condition+'/'+perp+'_'+condition+'_E_emd_spectra_bands_'+str(name)+'.csv')
                    fft_spectra_bands_df.to_csv('Spectra/'+perp+'/'+condition+'/'+perp+'_'+condition+'_E_fft_spectra_bands_'+str(name)+'.csv')

                    #holospectrum (produce a single average holo-spectrum)
                    if holosp:
                        holospectrum_avg = np.nanmean(holospectrum_array_avg[:, :, margin[0]:margin[1]], axis=2)
                        np.save('Spectra/'+perp+'/'+condition+'/'+perp+'_'+condition+'_E_'+name+'_holo.npy', holospectrum_avg)

                print(perp, condition, 'epochs done')

            art_free_perps = ['PCJ09', 'PFM82', 'PDH47', 'PLM37', 'PJS04', 'PVU29', 'PUA35', 'PTS72', 'P2M63', 'P5P11', 'PGH44', 'P4O85', 'PBA32']
            
            if perp in art_free_perps:
                print(perp, condition, 'starting full')

                if not os.path.exists('Spectra/'+perp+'/'+condition+'/'+perp+'_'+condition+'_F_emd_spectra_bands_Ppeak.csv'):

                    res_f = [10636,2995]
                    fmin = 0.1; fmax = 50
                    band_range_dict = {'slow': [0.1,1.5], 'delta': [1.5,4], 'theta': [4,8], 'alpha': [8,12], 'beta': [12,30], 'gamma': [30,50]}

                    holospectrum = np.nan * np.zeros((carrier_res, am_res, n_regions))

                    for res in res_f:
                        data = np.load('Data/Clean/Full/'+perp+'/'+condition+'/'+perp+'_'+condition+'_peak.npy')
                        name = 'peak'
                        if res == 2995:
                            data = data[:, 60*sfreq:(60*sfreq)+(60*sfreq)]
                            name = 'Ppeak'

                        freq_edges, freq_centres = emd.spectra.define_hist_bins(fmin, fmax, res, 'linear')

                        emd_spectra_df = pd.DataFrame(index=freq_centres, columns=np.arange(n_regions))
                        fft_spectra_df = pd.DataFrame(index=freq_centres, columns=np.arange(n_regions))
                        emd_spectra_bands_df = pd.DataFrame(index=band_range_dict.keys(), columns=np.arange(n_regions))
                        fft_spectra_bands_df = pd.DataFrame(index=band_range_dict.keys(), columns=np.arange(n_regions))

                        for region in range(n_regions):
                            data_region = data[region]
                            imfs = emd_imfs(data_region, sfreq=sfreq)
                            emd_spect, freq_centres = emd_spectra(imfs, None, None, mode='imfs', sfreq=sfreq, res=res, fmin=fmin, fmax=fmax, form='linear')
                            fft_spect, freqs = fft_spectra(data_region, sfreq=sfreq, fmin=fmin, fmax=fmax, form='linear')

                            if np.isnan(emd_spect).any():
                                emd_spect = fix_edge_case(emd_spect)

                            emd_spectra_df[region] = emd_spect
                            fft_spectra_df[region] = fft_spect
                            emd_spectra_bands_df[region] = get_bands(emd_spect, freq_centres, band_range_dict)
                            fft_spectra_bands_df[region] = get_bands(fft_spect, freqs, band_range_dict)

                            if holosp:
                                am_lf = .1; am_hf = 4
                                fcarrier, fam, holo = get_holo(imfs, sfreq, 
                                                                am_lf=am_lf, am_hf=am_hf, am_res=am_res, 
                                                                carrier_lf=carrier_lf, carrier_hf=carrier_hf, carrier_res=carrier_res)
                                holospectrum[:, :, region] = holo

                                # if fcarrer and fam are not saved then save
                                if not os.path.exists('Spectra/fcarrier.npy'):
                                    np.save('Spectra/fcarrier.npy', fcarrier)
                                    np.save('Spectra/fam.npy', fam)

                        if holosp:
                            holospectrum_avg = np.nanmean(holospectrum, axis=2)
                            np.save('Spectra/'+perp+'/'+condition+'/'+perp+'_'+condition+'_F_'+name+'_holo.npy', holospectrum)

                        #full (average over electrodes)
                        emd_spectra_avg = np.mean(emd_spectra_df, axis=1)
                        fft_spectra_avg = np.mean(fft_spectra_df, axis=1)
                        emd_spectra_avg_df = pd.DataFrame(index=freq_centres, columns=['avg'], data=emd_spectra_avg)
                        fft_spectra_avg_df = pd.DataFrame(index=freq_centres, columns=['avg'], data=fft_spectra_avg)
                        emd_spectra_avg_df.to_csv('Spectra/'+perp+'/'+condition+'/'+perp+'_'+condition+'_F_emd_spectra_'+str(name)+'.csv')
                        fft_spectra_avg_df.to_csv('Spectra/'+perp+'/'+condition+'/'+perp+'_'+condition+'_F_fft_spectra_'+str(name)+'.csv')

                        #bands
                        emd_spectra_bands_df.to_csv('Spectra/'+perp+'/'+condition+'/'+perp+'_'+condition+'_F_emd_spectra_bands_'+str(name)+'.csv')
                        fft_spectra_bands_df.to_csv('Spectra/'+perp+'/'+condition+'/'+perp+'_'+condition+'_F_fft_spectra_bands_'+str(name)+'.csv')

                    print(perp, condition, 'full done')

# run the parallel riddim
import joblib
from joblib import Parallel, delayed

num_cores = 19

Parallel(n_jobs=num_cores)(
    delayed(parallel_process)([perp]) for perp in perps
)

