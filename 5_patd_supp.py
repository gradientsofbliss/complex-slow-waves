# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

perps = ['PCJ09', 'PFM82', 'PDH47', 'PLM37', 'PJS04', 'PVU29', 'PUA35', 'PTS72', 'P2M63', 'P5P11', 'PGH44', 'P4O85', 'PBA32']
conditions = ['Rest', '5-MeO-DMT']
sfreq = 500
n_regions = 64

modes = [3, 4, 5]
modes_index = [m-1 for m in modes]
band_range_dict = {'slow': [0.1,1.5], 'delta': [1.5,4], 'theta': [4,8], 'alpha': [8,12], 'beta': [12,30], 'gamma': [30,50]}
band_symbol_dict = {'slow': (r'$s$'), 'delta': (r'$\delta$'), 'theta': (r'$\theta$'), 'alpha': (r'$\alpha$'), 'beta': (r'$\beta$'), 'gamma': (r'$\gamma$')}

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

def aug_wsr_test(array1, array2):
    '''
    Wilcoxon Signed Rank Test

    Parameters
    ----------
    array1: array
    array2: array
    
    array1 first so positive W-value mean array1 > array2

    '''
    import pingouin as pg

    result = pg.wilcoxon(array1, array2, tail='two-sided')

    w = result['W-val'].values[0]
    p = result['p-val'].values[0]
    rbc = result['RBC'].values[0] # matched pairs rank biserial correlation
    cles = result['CLES'].values[0] # common language effect size

    return w, p, rbc, cles
    
mode_dict = {1: 'Delta', 2: 'Slow', 3: 'Ultra-Slow'}

wave_dict = {0: 'Unknown',
             1: 'Stable Node',
             2: 'Stable Focus',
             3: 'Unstable Node',
             4: 'Unstable Focus',
             5: 'Saddle'
             }


abbrev_dict = {0: 'Other',
                1: 'S-Node',
                2: 'S-Focus',
                3: 'U-Node',
                4: 'U-Focus',
                5: 'Saddle'
                }

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
# # plot recurrence for each participant, mode, and condition
# from scipy.sparse import load_npz
# for mode in modes:
#     fig, ax = plt.subplots(2, len(perps), figsize=(2*len(perps), 2*2))
#     for i, perp in enumerate(perps):
#         for j, condition in enumerate(conditions):
#             sig_mat = load_npz(f'flow_fields/{perp}/{condition}/mode_{mode}/sig_mat.npz')
#             sig_mat = sig_mat.toarray()
#             sig_mat = sig_mat.astype(int)
#             n_nodes = sig_mat.shape[0]
#             ax[j,i].imshow(sig_mat, cmap='Greys', aspect='auto')
#             ax[j,i].set_title(f'S{i+1} {condition}')
#             ax[j,i].set_xticks([])
#             ax[j,i].set_yticks([]) 
#     plt.tight_layout()
# # wayyyyyyyyyyyyy too much memory

# %%
# instead of below, should really be doing it all in one and correcting accross



# %%
for mode in modes:
    rest_pers = np.zeros((len(perps), len(abbrev_dict)))
    five_pers = np.zeros((len(perps), len(abbrev_dict)))
    rest_prop = np.zeros((len(perps), len(abbrev_dict)))
    five_prop = np.zeros((len(perps), len(abbrev_dict)))
    for perp in perps:
        for condition in conditions:
            persistence = np.load(f'flow_fields/{perp}/{condition}/mode_{mode}/dynamics/persistence.npy', allow_pickle=True)
            proportion = np.load(f'flow_fields/{perp}/{condition}/mode_{mode}/dynamics/proportions.npy', allow_pickle=True)
            pers = np.zeros(len(abbrev_dict))
            prop = np.zeros(len(abbrev_dict))
            for key, value in persistence.item().items():
                pers[key] = value
            for key, value in proportion.item().items():
                prop[key] = value
            if condition == 'Rest':
                rest_pers[perps.index(perp), :] = pers
                rest_prop[perps.index(perp), :] = prop
            else:
                five_pers[perps.index(perp), :] = pers
                five_prop[perps.index(perp), :] = prop

    # save
    np.save(f'flow_fields/persistence_mode_{mode}_rest.npy', rest_pers)
    np.save(f'flow_fields/persistence_mode_{mode}_5.npy', five_pers)
    np.save(f'flow_fields/proportion_mode_{mode}_rest.npy', rest_prop)
    np.save(f'flow_fields/proportion_mode_{mode}_5.npy', five_prop)

# do stats
alt_dyn_stats_df = pd.DataFrame(columns=['mode', 'pattern', 'measure',
                                         't', 'p', 'cil', 'ciu', 'd', 'bf', 'dof', 'power'])
for mode in modes:
    rest_pers = np.load(f'flow_fields/persistence_mode_{mode}_rest.npy')
    five_pers = np.load(f'flow_fields/persistence_mode_{mode}_5.npy')
    rest_prop = np.load(f'flow_fields/proportion_mode_{mode}_rest.npy')
    five_prop = np.load(f'flow_fields/proportion_mode_{mode}_5.npy')
    for i in range(len(abbrev_dict)):
        rest = rest_pers[:, i]
        five = five_pers[:, i]
        t, p, cil, ciu, d, bf, dof, power = aug_t_test(five, rest)
        alt_dyn_stats_df = alt_dyn_stats_df.append({'mode': mode, 'pattern': abbrev_dict[i], 'measure': 'persistence',
                                              't': t, 'p': p, 'cil': cil, 'ciu': ciu, 'd': d, 'bf': bf, 'dof': dof, 'power': power}, ignore_index=True)
        rest = rest_prop[:, i]
        five = five_prop[:, i]
        t, p, cil, ciu, d, bf, dof, power = aug_t_test(five, rest)
        alt_dyn_stats_df = alt_dyn_stats_df.append({'mode': mode, 'pattern': abbrev_dict[i], 'measure': 'proportion',
                                                't': t, 'p': p, 'cil': cil, 'ciu': ciu, 'd': d, 'bf': bf, 'dof': dof, 'power': power}, ignore_index=True)

alt_dyn_stats_df['p_bh'] = multi.multipletests(alt_dyn_stats_df['p'], method='fdr_bh')[1]
alt_dyn_stats_df.to_csv('flow_fields/alt_dyn_stats.csv', index=False)

alt_dyn_stats_df

# %%
for mode in modes:
    rest_tpm = np.zeros((len(perps), len(abbrev_dict), len(abbrev_dict)))
    five_tpm = np.zeros((len(perps), len(abbrev_dict), len(abbrev_dict)))
    for perp in perps:
        for condition in conditions:
            tpm = np.load(f'flow_fields/{perp}/{condition}/mode_{mode}/dynamics/tpm.npy', allow_pickle=True)
            # tpm is a matrix of size len(abbrev_dict) x len(abbrev_dict)
            if condition == 'Rest':
                rest_tpm[perps.index(perp), :, :] = tpm
            else:
                five_tpm[perps.index(perp), :, :] = tpm
    
    # save
    np.save(f'flow_fields/tpm_mode_{mode}_rest.npy', rest_tpm)
    np.save(f'flow_fields/tpm_mode_{mode}_5.npy', five_tpm)

# do stats
tpm_stats_df = pd.DataFrame(columns=['mode', 'pattern1', 'pattern2', 't', 'p', 'cil', 'ciu', 'd', 'bf', 'dof', 'power'])
for mode in modes:
    rest_tpm = np.load(f'flow_fields/tpm_mode_{mode}_rest.npy')
    five_tpm = np.load(f'flow_fields/tpm_mode_{mode}_5.npy')
    for i in range(len(abbrev_dict)):
        for j in range(len(abbrev_dict)):
            rest = rest_tpm[:, i, j]
            five = five_tpm[:, i, j]
            t, p, cil, ciu, d, bf, dof, power = aug_t_test(five, rest)
            tpm_stats_df = tpm_stats_df.append({'mode': mode, 'pattern1': abbrev_dict[i], 'pattern2': abbrev_dict[j], 't': t, 'p': p, 'cil': cil, 'ciu': ciu, 'd': d, 'bf': bf, 'dof': dof, 'power': power}, ignore_index=True)
# if nan set to 1
tpm_stats_df.fillna(1, inplace=True)
# correct p-values
tpm_stats_df['p_bh'] = multi.multipletests(tpm_stats_df['p'], method='fdr_bh')[1]

# plot a difference matrix for each mode with the significant differences

for mode in modes:
    # make a matrix of the tpm t values
    diff_tpm = np.zeros((len(abbrev_dict), len(abbrev_dict)))
    for i in range(len(abbrev_dict)):
        for j in range(len(abbrev_dict)):
            t = tpm_stats_df[(tpm_stats_df['mode'] == mode) & (tpm_stats_df['pattern1'] == abbrev_dict[i]) & (tpm_stats_df['pattern2'] == abbrev_dict[j])]['t'].values[0]
            diff_tpm[i, j] = t
    np.fill_diagonal(diff_tpm, 0)
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.5))
    max = np.percentile(np.abs(diff_tpm), 95)
    img = ax.imshow(diff_tpm, vmin=-6, vmax=6, cmap='PRGn_r', aspect='auto')
    ax.set_xticks(np.arange(len(abbrev_dict)))
    ax.set_yticks(np.arange(len(abbrev_dict)))
    ax.set_xticklabels(abbrev_dict.values())
    ax.set_yticklabels(abbrev_dict.values())
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha='right', rotation_mode="anchor")
    # add labels
    plt.xlabel(r'$X_{t+1}$')
    plt.ylabel(r'$X_t$')
    #plt.title(f'Mode {mode}')
    # add significant differences
    for i in range(len(abbrev_dict)):
        for j in range(len(abbrev_dict)):
            p = tpm_stats_df[(tpm_stats_df['mode'] == mode) & (tpm_stats_df['pattern1'] == abbrev_dict[i]) & (tpm_stats_df['pattern2'] == abbrev_dict[j])]['p_bh'].values[0]
            if p < 0.05:
                ax.text(j, i+0.2, '*', ha='center', va='center', color='white', fontsize=30)
    # plt.colorbar(img, ax=ax, label=r'$\Delta P(X_{t+1} = j \mid X_t = i)$')
    plt.colorbar(img, ax=ax, label=r'$T$')
    plt.show()





