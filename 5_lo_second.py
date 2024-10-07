# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import warnings
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

import os
print(os.getcwd())

full_perps = ['PCJ09', 'PDH47', 'PLM37', ' PVU29', 'PTS72', 'P2M63', 'P5P11', 'PBA32']
peak_perps = ['PCJ09', 'PFM82', 'PDH47', 'PLM37', 'PJS04', 'PVU29', 'PUA35', 'PTS72', 'P2M63', 'P5P11', 'PGH44', 'P4O85', 'PBA32']
seg_perps = ['PCJ09', 'PFM82', 'PDH47', 'PLM37', 'PJS04', 'PVU29', 'PUA35', 'PTS72', 'PXH23', 'PQS29', 'P2M63', 'P5P11', 'PGH44', 'P4O85', 'P3N74', 'PBA32', 'PDL71', 'P7R50', 'PFQ62']

perps = peak_perps

conditions = ['Rest', '5-MeO-DMT']
sfreq = 500
n_regions = 64

electrode_locs_dict = np.load('electrode_locs.npy', allow_pickle=True)
ch_pos = electrode_locs_dict.item()
ch_pos = np.array(list(ch_pos.values()))
ch_pos_x = ch_pos[:,0]
ch_pos_y = ch_pos[:,1]

def aug_t_test(array1, array2):
    '''
    T-test

    Parameters
    ----------
    array1: array
    array2: array
    
    array1 first so positive T mean array1 > array2

    '''
    import pingouin as pg

    result = pg.ttest(array1, array2, paired=True)
    
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

# %%
from lo_library import *

# %%
n_eigen = 10

run_analysis = 'True'

if run_analysis:

    five_eigen = np.zeros((len(perps), n_eigen))
    rest_eigen = np.zeros((len(perps), n_eigen))

    for i, perp in enumerate(perps):
        for condition in conditions:

            if not os.path.exists('low_dim/'+perp+'/'+condition):
                os.makedirs('low_dim/'+perp+'/'+condition)

            signals = np.load('Data/Clean/Full/'+perp+'/'+condition+'/'+perp+'_'+condition+'_peak.npy')
            signals = signals[:, 60*sfreq:(60*sfreq)+(60*sfreq)]
            signals = amp_env(signals)

            eigenvectors, eigenvalues = pca_c(signals, n_eigen, method='svd')
            np.save('low_dim/'+perp+'/'+condition+'/pca_eigenvalues.npy', eigenvalues)

            if condition == 'Rest':
                rest_eigen[i] = eigenvalues
            else:
                five_eigen[i] = eigenvalues

stats_df = pd.DataFrame(columns=['measure', 'T', 'p', 'cil', 'ciu', 'd', 'bf', 'dof', 'power', 'p_bh'])
for i in range(n_eigen):
    t, p, cil, ciu, d, bf, dof, power = aug_t_test(five_eigen[:, i], rest_eigen[:, i])
    stats_df = stats_df.append({'measure': 'eigen_'+str(i+1), 'T': t, 'p': p, 'cil': cil, 'ciu': ciu, 'd': d, 'bf': bf, 'dof': dof, 'power': power}, ignore_index=True)
stats_df['p_bh'] = multi.multipletests(stats_df['p'], method='fdr_bh')[1]
stats_df.to_csv('low_dim/pca_stats.csv')

mean_five = np.mean(five_eigen, axis=0)
mean_rest = np.mean(rest_eigen, axis=0)
se_five = stats.sem(five_eigen, axis=0)
se_rest = stats.sem(rest_eigen, axis=0)

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(np.arange(1, n_eigen+1), mean_rest, label='Rest', color='green')
#ax.errorbar(np.arange(1, n_eigen+1), mean_rest, yerr=se_rest, fmt='o', color='green', capsize=5)
ax.fill_between(np.arange(1, n_eigen+1), mean_rest-se_rest, mean_rest+se_rest, color='green', alpha=0.1)
ax.plot(np.arange(1, n_eigen+1), mean_five, label='5-MeO-DMT', color='purple')
# ax.errorbar(np.arange(1, n_eigen+1), mean_five, yerr=se_five, fmt='o', color='purple', capsize=5)
ax.fill_between(np.arange(1, n_eigen+1), mean_five-se_five, mean_five+se_five, color='purple', alpha=0.1)
ax.set_xlabel('Eigenvector')
ax.set_ylabel('Eigenvalue')
ax.set_yscale('log', base=10)
plt.legend(frameon=False)
sns.despine()

star_pos = np.median([mean_five-se_five, mean_rest], axis=0)

for i in range(n_eigen):
    if stats_df['p_bh'][i] < 0.05:
        # want it to be mid way between the error bars
        plt.text(i+1, star_pos[i],
                 '*', ha='center', va='center', fontsize=20, color='black')
    elif stats_df['p'][i] < 0.1:
        plt.text(i+1, star_pos[i],
                 '*', ha='center', va='center', fontsize=20, color='grey')

# %%
# write stats for PC1
t = write_scientific(stats_df['T'][0])
p = write_scientific(stats_df['p_bh'][0])
b = write_scientific(stats_df['bf'][0])
d = write_scientific(stats_df['d'][0])
print(f'$T=${t}, $p_{{FDR}}=${p}, $BF_{{10}}=${b}, $d=${d}')

# %%
timescales_df = pd.read_csv('low_dim/timescale.csv')

timescales_stats_df = pd.DataFrame(columns=['measure', 'T', 'p', 'cil', 'ciu', 'd', 'bf', 'dof', 'power', 'p_bh'])

for measure in ['max_lambda', 'ami']:
    t, p, cil, ciu, d, bf, dof, power = aug_t_test(timescales_df[timescales_df['condition'] == '5-MeO-DMT'][measure], timescales_df[timescales_df['condition'] == 'Rest'][measure])
    timescales_stats_df = timescales_stats_df.append({'measure': measure, 'T': t, 'p': p, 'cil': cil, 'ciu': ciu, 'd': d, 'bf': bf, 'dof': dof, 'power': power}, ignore_index=True)
timescales_stats_df['p_bh'] = multi.multipletests(timescales_stats_df['p'], method='fdr_bh')[1]
timescales_stats_df.to_csv('low_dim/timescale_stats.csv')

for measure in ['max_lambda', 'ami']:
    fig, ax = plt.subplots(1, 1, figsize=(2, 5))
    plt.xlim(-0, 1)
    for j in range(len(perps)):
        ax.plot([0, 1], [timescales_df[timescales_df['perp'] == perps[j]][measure].values[0], timescales_df[timescales_df['perp'] == perps[j]][measure].values[1]], color='black', alpha=0.1)
    sns.pointplot(x='condition', y=measure, data=timescales_df, ax=ax, dodge=True,
                    palette={"Rest": "green", "5-MeO-DMT": "purple"}, ci=95, markers='d')
    rest_mean = timescales_df[timescales_df['condition'] == 'Rest'][measure].mean()
    five_mean = timescales_df[timescales_df['condition'] == '5-MeO-DMT'][measure].mean()
    from matplotlib.colors import LinearSegmentedColormap
    cdict = {'red':   [[0.0, 0.0, 0.0],
                          [1.0, 0.5, 0.5]],
                 'green': [[0.0, 0.5, 0.5],
                          [1.0, 0.0, 0.0]],
                 'blue':  [[0.0, 0.0, 0.0],
                          [1.0, 0.5, 0.5]]}
    custom_cmap = LinearSegmentedColormap('custom_cmap', cdict)
    x = np.linspace(0, 1, 1000)
    colors = custom_cmap(x)
    m = (five_mean - rest_mean) / 1
    for i in range(x.shape[0]-1):
        ax.plot([x[i], x[i+1]], [m*x[i] + rest_mean, m*x[i+1] + rest_mean], color=colors[i], linewidth=2)
    # mean of the means
    max_y = np.mean([rest_mean, five_mean])
    if timescales_stats_df[timescales_stats_df['measure'] == measure]['p_bh'].values[0] < 0.001:
        plt.text(0.5, max_y,
                 '***', ha='center', va='center', fontsize=20, color='k')
    elif timescales_stats_df[timescales_stats_df['measure'] == measure]['p_bh'].values[0] < 0.01:
        plt.text(.5, max_y,
                 '**', ha='center', va='center', fontsize=20, color='k')
    elif timescales_stats_df[timescales_stats_df['measure'] == measure]['p_bh'].values[0] < 0.05:
        plt.text(0.5, max_y,
                 '*', ha='center', va='center', fontsize=20, color='k')
    if measure == 'max_lambda':
        ax.set_ylabel(r'Seperation rate ($\lambda_{max}$)')
    else:
        ax.set_ylabel('Intrinsic timescale (ms)')
    ax.set_xlabel('')
    ax.set_xticklabels(['Rest', '5-MeO'])
    sns.despine()

timescales_stats_df


# %%
for result in ['ami', 'max_lambda']:
    print(result)
    print('Rest:', round(timescales_df[timescales_df['condition'] == 'Rest'][result].mean(), 1), r'$\pm$', round(stats.sem(timescales_df[timescales_df['condition'] == 'Rest'][result]), 1))
    print('5-MeO:', round(timescales_df[timescales_df['condition'] == '5-MeO-DMT'][result].mean(), 1), r'$\pm$', round(stats.sem(timescales_df[timescales_df['condition'] == '5-MeO-DMT'][result]), 1))

    t = write_scientific(timescales_stats_df[timescales_stats_df['measure'] == result]['T'].values[0])
    p = write_scientific(timescales_stats_df[timescales_stats_df['measure'] == result]['p_bh'].values[0])
    b = write_scientific(timescales_stats_df[timescales_stats_df['measure'] == result]['bf'].values[0])
    d = write_scientific(timescales_stats_df[timescales_stats_df['measure'] == result]['d'].values[0])
    print(f'$T=${t}, $p_{{FDR}}=${p}, $BF_{{10}}=${b}, $d=${d}.')


# %%
ami_measures = ['ami_s2', 'ami_s3', 'ami_p50']

if not os.path.exists('low_dim/timescale_ext.csv'):
    timescale_ext_df = pd.DataFrame(columns=['perp', 'condition', 'ami_s2', 'ami_s3', 'ami_p50'])

    for perp in perps:
        for condition in conditions:
            amis = {}
            am = [2,3,50]
            amis = {a: [] for a in am}

            signals = np.load('Data/Clean/Full/'+perp+'/'+condition+'/'+perp+'_'+condition+'_peak.npy')
            signals = signals[:, 60*sfreq:(60*sfreq)+(60*sfreq)]
            signals = amp_env(signals)

            for region in range(n_regions):
                signals = np.load('Data/Clean/Full/'+perp+'/'+condition+'/'+perp+'_'+condition+'_peak.npy')
                for a in am:
                    if a != 50:
                        ami = auto_mi(signals[region, :], sfreq, option='minima', thresh=a, t_max=None)
                    else:
                        ami = auto_mi(signals[region, :], sfreq, option='percentile', thresh=a, t_max=None)
                    amis[a].append(ami)
            timescale_ext_df = timescale_ext_df.append({'perp': perp, 'condition': condition, 
                                                'ami_s2': np.nanmean(np.array(amis[2])),
                                                'ami_s3': np.nanmean(np.array(amis[3])), 
                                                'ami_p50': np.nanmean(np.array(amis[50]))}, 
                                                ignore_index=True)
            timescale_ext_df.to_csv('low_dim/timescale_ext.csv', index=False)

else:
    timescale_ext_df = pd.read_csv('low_dim/timescale_ext.csv')

timescales_ext_stats_df = pd.DataFrame(columns=['measure', 'T', 'p', 'cil', 'ciu', 'd', 'bf', 'dof', 'power'])
for measure in ami_measures:
    t, p, cil, ciu, d, bf, dof, power = aug_t_test(timescale_ext_df[timescale_ext_df['condition'] == '5-MeO-DMT'][measure], timescale_ext_df[timescale_ext_df['condition'] == 'Rest'][measure])
    timescales_ext_stats_df = timescales_ext_stats_df.append({'measure': measure, 'T': t, 'p': p, 'cil': cil, 'ciu': ciu, 'd': d, 'bf': bf, 'dof': dof, 'power': power}, ignore_index=True)

timescales_ext_stats_df

# %%
n_ms = len(ami_measures)
fig, ax = plt.subplots(1, n_ms, figsize=(n_ms*2.2, 3.5))

for pl, a in enumerate(ami_measures):
    ax[pl].set_xlim(-0, 1)
    for j in range(len(perps)):
        ax[pl].plot([0, 1], [timescale_ext_df[timescale_ext_df['perp'] == perps[j]][a].values[0], timescale_ext_df[timescale_ext_df['perp'] == perps[j]][a].values[1]], color='black', alpha=0.1)
    sns.pointplot(x='condition', y=a, data=timescale_ext_df, ax=ax[pl], dodge=True,
                    palette={"Rest": "green", "5-MeO-DMT": "purple"}, ci=95, markers='d')
                    
    rest_mean = timescale_ext_df[timescale_ext_df['condition'] == 'Rest'][a].mean()
    five_mean = timescale_ext_df[timescale_ext_df['condition'] == '5-MeO-DMT'][a].mean()
    from matplotlib.colors import LinearSegmentedColormap
    cdict = {'red':   [[0.0, 0.0, 0.0],
                          [1.0, 0.5, 0.5]],
                 'green': [[0.0, 0.5, 0.5],
                          [1.0, 0.0, 0.0]],
                 'blue':  [[0.0, 0.0, 0.0],
                          [1.0, 0.5, 0.5]]}
    custom_cmap = LinearSegmentedColormap('custom_cmap', cdict)
    x = np.linspace(0, 1, 1000)
    colors = custom_cmap(x)
    m = (five_mean - rest_mean) / 1
    for i in range(x.shape[0]-1):
        ax[pl].plot([x[i], x[i+1]], [m*x[i] + rest_mean, m*x[i+1] + rest_mean], color=colors[i], linewidth=2)
    max = timescale_ext_df[timescale_ext_df['condition'] == '5-MeO-DMT'][a].max()
    p = timescales_ext_stats_df[timescales_ext_stats_df['measure'] == a]['p'].values[0]
    if p < 0.001:
        ax[pl].text(0.5, max,
                 '***', ha='center', va='center', fontsize=20, color='k')
    elif p < 0.01:
        ax[pl].text(.5, max,
                 '**', ha='center', va='center', fontsize=20, color='k')
    elif p < 0.05:
        ax[pl].text(0.5, max,
                 '*', ha='center', va='center', fontsize=20, color='k')
    if pl == 0:
        ax[pl].set_ylabel('Intrinsic timescale (ms)')
    else:
        ax[pl].set_ylabel('')
    ax[pl].set_xlabel('')
    ax[pl].set_xticklabels(['Rest', '5-MeO'])
    sns.despine()

    ax[pl].text(-0.5, 1, chr(97+pl), transform=ax[pl].transAxes,
            size=15, weight='bold')

plt.tight_layout()

# save
fig.savefig('MS1_Figs/Supp_Fig6.png', dpi=500)
    

# %%


# %%


# %%



