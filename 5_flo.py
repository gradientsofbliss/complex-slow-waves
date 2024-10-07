import numpy as np
import pandas as pd
import os
from scipy import sparse

import mne
import emd

from flo_library import *

def parallel_process(perp):

    conditions = ['Rest', '5-MeO-DMT']
    sfreq = 500
    n_regions = 64

    raw = mne.io.read_raw_edf('raw.edf', preload=True)
    raw.rename_channels(lambda x: x[:-4]); raw.rename_channels(lambda x: x[4:])
    ch_names = raw.info['ch_names']

    electrode_locs_dict = np.load('electrode_locs.npy', allow_pickle=True)
    ch_pos = electrode_locs_dict.item()
    ch_pos = np.array(list(ch_pos.values()))

    max_imfs = 6
    modes = [3, 4, 5]
    modes_index = [m-1 for m in modes]

    res = 32
    nas_in = 0.35
    d0 = 0.8*nas_in # scalp diameter
    xgrid = np.mgrid[-d0/2:d0/2:res*1j, -d0/2:d0/2:res*1j] # grid for interpolation
    Y, X = xgrid
    circle = (X**2 + Y**2) < (d0/2)**2 # scalp circle
    xflat = xgrid.reshape(2, -1).T # flattened grid
    R = np.sqrt(X**2 + Y**2)
    mask = R < 1

    wave_dict = {0: 'Unknown',
             1: 'Stable Node',
             2: 'Stable Focus',
             3: 'Unstable Node',
             4: 'Unstable Focus',
             5: 'Saddle'
             }

    for condition in conditions:

        if not os.path.exists('modes/'+perp+'/'+condition):
            os.makedirs('modes/'+perp+'/'+condition)
        if not os.path.exists('flow_fields/'+perp+'/'+condition):
           os.makedirs('flow_fields/'+perp+'/'+condition)
        

        signals = np.load('Data/Clean/Full/'+perp+'/'+condition+'/'+perp+'_'+condition+'_peak.npy')

        # get imfs

        if not os.path.exists('modes/'+perp+'/'+condition+'/freqs.csv'):
            freq_df = pd.DataFrame(columns=np.arange(n_regions), index=np.arange(max_imfs))
        for region in range(n_regions):
            kwargs = {'path': 'modes/'+perp+'/'+condition+'/electrode_'+str(region)+'/'}
            if not os.path.exists(kwargs['path']):
                os.makedirs(kwargs['path'])
            if not os.path.exists(kwargs['path']+'IA.npy'):
                timeseries = signals[region]
                imfs, IP, IF, IA = emd_imfs(timeseries, sfreq, max_imfs=max_imfs, save=True, **kwargs)
                for mode in np.arange(max_imfs):
                    freq_df.loc[mode, region] = np.mean(IF[:, mode])
        if not os.path.exists('modes/freqs/'+perp+'/'+condition+'/freqs.csv'):
            if not os.path.exists('modes/freqs/'+perp+'/'+condition):
                os.makedirs('modes/freqs/'+perp+'/'+condition)
            freq_df.to_csv('modes/freqs/'+perp+'/'+condition+'/freqs.csv')
        
        # smooth instantaneous properties

        if not os.path.exists('modes/'+perp+'/'+condition+'/mode_'+str(modes[-1])+'/smoothed_IA_mode'+str(modes[-1])+'.npy'):

            for prop in ['IA','IP']:
                for mode in modes:
                    mode_index = mode-1
                    for region in range(n_regions):
                        IP = np.load('modes/'+perp+'/'+condition+'/electrode_'+str(region)+'/'+prop+'.npy')
                        IP = IP[60*sfreq:(60*sfreq)+(60*sfreq), mode_index] # from 1.5 to 2.5 minutes
                        if region == 0:
                            IP_all = IP
                        else:
                            IP_all = np.dstack((IP_all, IP))
                    out, X, Y = smooth_spatial(IP_all, ch_names, sfreq, ch_pos, input=prop,
                                                    res=32, kernel_size=1, 
                                                    out_wrapped=True, smooth_wrapped=True,
                                                    nas_in=0.35, raw=raw)
                    if not os.path.exists('modes/'+perp+'/'+condition+'/mode_'+str(mode)):
                        os.makedirs('modes/'+perp+'/'+condition+'/mode_'+str(mode))
                    np.save('modes/'+perp+'/'+condition+'/mode_'+str(mode)+'/smoothed_'+prop+'_mode'+str(mode)+'.npy', out)

                    if not os.path.exists('flow_fields/X.npy'):
                        np.save('flow_fields/X.npy', X)
                        np.save('flow_fields/Y.npy', Y)

        # compute velocity fields and derived quantities

        for mode in modes:

            if not os.path.exists('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)):
                os.makedirs('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode))

            vel_go = True

            if vel_go:

                if not os.path.exists('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/pts.npy'):

                    IA = np.load('modes/'+perp+'/'+condition+'/mode_'+str(mode)+'/smoothed_IA_mode'+str(mode)+'.npy')

                    u, v, speed = get_velocity_field(IA, sfreq)
                    del IA
                    np.save('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/u.npy', u)
                    np.save('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/v.npy', v)
                    np.save('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/speed.npy', speed)

                    phi = avg_norm_vel(u, v)
                    np.save('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/phi.npy', phi)

                    H = field_hetero(speed)
                    np.save('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/H.npy', H)
                    del speed, H

                    pts = find_singularities_divergence(u, v, mask=mask, nas_in=0.35, upsampling=4,
                                            extent=[-1, 1], nlevels=10, robust_levels=True,
                                            robust_th=.01, upsample_vel=False)
                    np.save('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/pts.npy', pts)
                else:
                    pts = np.load('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/pts.npy', allow_pickle=True)
                    u = np.load('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/u.npy')
                    v = np.load('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/v.npy')
                    phi = np.load('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/phi.npy')
                
                idxs = np.arange(len(u))
                times = ((idxs+1) / sfreq) * 1000 # 500hz sampling rate so timepoints are 2ms apart
                singularities_df = sings_to_df(pts, idxs, times, phi, extent=[-1, 1], res=32)
                del pts
                singularities_df = singularity_taxonomy(u, v, singularities_df)
                singularities_df.to_csv('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/singularities_df.csv')

                if not os.path.exists('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/dynamics'):
                    os.makedirs('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/dynamics')

                wprops_df = wave_properites(singularities_df, wave_dict, sfreq)
                wprops_df.to_csv('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/dynamics/wprops_df.csv')

                dom_waves = wave_dominance(phi, singularities_df)
                np.save('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/dynamics/dom_waves.npy', dom_waves)
                proportions, persistence, tpm = wave_dom_dynamics(dom_waves, wave_dict, sfreq)
                np.save('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/dynamics/proportions.npy', proportions)
                np.save('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/dynamics/persistence.npy', persistence)
                np.save('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/dynamics/tpm.npy', tpm)

        for mode in modes:

            coh_go = False

            if coh_go:

                IP = np.load('modes/'+perp+'/'+condition+'/mode_'+str(mode)+'/smoothed_IA_mode'+str(mode)+'.npy')
                local_radi = np.arange(2, 14, 1) # stopping it before 15 because by then we are using < 50% of the scalp
                base_columns = ['perp', 'condition', 'mode', 'Global Coherence', 'Global Metastability']
                columns = base_columns + \
                    [f'Local Coherence {r}' for r in local_radi] + \
                    [f'Local Metastability {r}' for r in local_radi] + \
                    [f'TP-V Coherence r{r}' for r in local_radi] + \
                    [f'TP-V Metastability r{r}' for r in local_radi]
                coherence_df = pd.DataFrame(columns=columns)
                R = kuramoto_order_parameter(IP)
                np.save('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/R.npy', R)
                global_coh = global_coherence(R)
                global_met = global_metastability(R)
                row_data = {'perp': perp, 'condition': condition,'mode': mode,
                            'Global Coherence': global_coh,'Global Metastability': global_met}
                for r in local_radi:
                    local_coh, local_met, topvar_coh, topvar_met = local_kuramoto(IP, r)
                    row_data[f'Local Coherence {r}'] = local_coh
                    row_data[f'Local Metastability {r}'] = local_met
                    row_data[f'TP-V Coherence r{r}'] = topvar_coh
                    row_data[f'TP-V Metastability r{r}'] = topvar_met
                coherence_df = coherence_df.append(row_data, ignore_index=True)
                coherence_df.to_csv('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/coherence_df.csv')
                del IP, local_radi, base_columns, columns, coherence_df, R, global_coh, global_met, row_data

        for mode in modes:

            align_go = False

            if align_go:
                
                if not os.path.exists('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/sig_mat.npz'):
                    u = np.load('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/u.npy')
                    v = np.load('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/v.npy')
                    u = u.reshape(-1, 2, 32, 32); v = v.reshape(-1, 2, 32, 32)
                    u = u.mean(axis=1); v = v.mean(axis=1) # downsample to 250Hz
                    dot_mat = velocity_alignment(u, v)
                    #np.save('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/dot_mat.npy', dot_mat)
                    out, max_perm = shuffle_velocity_align(u, v, flatten=True, nperm=100)
                    np.save('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/max_perm.npy', max_perm)
                    sig_mat = thresh_matrix(dot_mat, max_perm)
                    #np.save('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/sig_mat.npy', sig_mat)
                    sparse.save_npz('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/sig_mat.npz', sparse.csr_matrix(sig_mat))
                    del dot_mat
                else:
                    sig_mat = sparse.load_npz('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/sig_mat.npz').toarray()
                    max_perm = np.load('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/max_perm.npy')
                
                alignment_df = pd.DataFrame(columns=['perp', 'condition', 'mode', 'max_perm', 
                                                     'ge', 'n_coms', 'mod', 'negentropy'])
                G = matrix_to_graph(sig_mat)
                degree_hist = degree_dist(G)
                negentropy = dist_negentropy(degree_hist)
                ge = global_efficiency(G)
                comms = community_detection(G)
                n_coms = len(comms)
                mod = modularity(G, comms)
                np.save('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/degree_hist.npy', degree_hist)
                alignment_df = alignment_df.append({'perp': perp, 'condition': condition, 'mode': mode, 'max_perm': max_perm, 
                                                    'ge': ge, 'n_coms': n_coms, 'mod': mod, 'negentropy': negentropy}, ignore_index=True)
                alignment_df.to_csv('flow_fields/'+perp+'/'+condition+'/mode_'+str(mode)+'/alignment_df.csv')
                del sig_mat, max_perm, alignment_df, G, degree_hist, negentropy, ge, comms, n_coms, mod


# run the parallel riddim
import joblib
from joblib import Parallel, delayed

num_cores = 13

perps = ['PCJ09', 'PFM82', 'PDH47', 'PLM37', 'PJS04', 'PVU29', 'PUA35', 'PTS72', 'P2M63', 'P5P11', 'PGH44', 'P4O85', 'PBA32']

Parallel(n_jobs=num_cores)(
    delayed(parallel_process)(perp) for perp in perps
)


            









