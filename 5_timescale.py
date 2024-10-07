import numpy as np
import pandas as pd
import os
import itertools
from lo_library import *
from scipy.stats import zscore

timescale_df = pd.DataFrame(columns=['perp', 'condition', 'max_lambda', 'ami'])
if not os.path.exists('low_dim'):
    os.makedirs('low_dim')
    timescale_df.to_csv('low_dim/timescale.csv', index=False)
timescale_df.to_csv('low_dim/timescale.csv', index=False)

def parallel_process(perp):

    conditions = ['Rest', '5-MeO-DMT']
    sfreq = 500
    n_regions = 64

    for condition in conditions:

        print(perp, condition)

        if not os.path.exists('low_dim/'+perp+'/'+condition):
            os.makedirs('low_dim/'+perp+'/'+condition)

        signals = np.load('Data/Clean/Full/'+perp+'/'+condition+'/'+perp+'_'+condition+'_peak.npy')
        signals = signals[:, 60*sfreq:(60*sfreq)+(60*sfreq)]
        signals = amp_env(signals)

        lyaps = []
        amis = []
        for region in range(n_regions):
            lyap = max_lambda(signals[region, :], sfreq)
            ami = auto_mi(signals[region, :], sfreq, option='minima', thresh=1)
            lyaps.append(lyap)
            amis.append(ami)
        lyaps = np.array(lyaps)
        amis = np.array(amis)
        np.save('low_dim/'+perp+'/'+condition+'/lyaps.npy', lyaps)
        np.save('low_dim/'+perp+'/'+condition+'/amis.npy', amis)

        timescale_df = pd.read_csv('low_dim/timescale.csv')
        timescale_df = timescale_df.append({'perp': perp, 'condition': condition, 'max_lambda': np.nanmean(lyaps), 'ami': np.nanmean(amis)}, ignore_index=True)
        timescale_df.to_csv('low_dim/timescale.csv', index=False)

# run the parallel riddim
import joblib
from joblib import Parallel, delayed

num_cores = 13

perps = ['PCJ09', 'PFM82', 'PDH47', 'PLM37', 'PJS04', 'PVU29', 'PUA35', 'PTS72', 'P2M63', 'P5P11', 'PGH44', 'P4O85', 'PBA32']

Parallel(n_jobs=num_cores)(
    delayed(parallel_process)(perp) for perp in perps
)