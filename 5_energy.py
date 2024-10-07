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

from lo_library import *

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

# %%
start = 10
t_max = 250 # in samples
t_max_ = t_max + start
inc = 0.2
ds = np.arange(1,5+inc,inc)
ntimes=200

# %%
# make a histogram of MSDs

run_hist = 'True'
top = 30
bins = 100

if run_hist:
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    hist_avg = np.zeros(bins)
    for perp in perps:
        for condition in conditions:
            signals = np.load('Data/Clean/Full/'+perp+'/'+condition+'/'+perp+'_'+condition+'_peak.npy')
            signals = signals[:, 60*sfreq:(60*sfreq)+(60*sfreq)]
            signals = amp_env(signals)
            signals = stats.zscore(signals, axis=1)

            spike_list = np.linspace(0, (signals.shape[1]-(t_max+start)), 200, dtype=int)

            msd_matrix = np.zeros((len(spike_list), t_max))

            for i, spike in enumerate(spike_list):
                msd_ = msd(spike, start, t_max_, signals)
                msd_matrix[i] = msd_

            # 0 to 30 hist
            msd_f = msd_matrix.flatten()
            hist = np.histogram(msd_f, bins=100, range=(0,top), density=True)
            hist_avg += hist[0]

    hist_avg = hist_avg/len(perps)/len(conditions)
    ax.plot(hist[1][:-1], hist_avg, label=perp+' '+condition, color='black')
    ax.axvline(np.max(ds), color='red', linestyle='--')
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.set_xlabel('MSD')
    ax.set_ylabel('Frequency')
    sns.despine()


# %%

run_analysis = 'True'

if run_analysis:

    for perp in perps:
        for condition in conditions:
            signals = np.load('Data/Clean/Full/'+perp+'/'+condition+'/'+perp+'_'+condition+'_peak.npy')
            signals = signals[:, 60*sfreq:(60*sfreq)+(60*sfreq)]
            signals = amp_env(signals)
            signals = stats.zscore(signals, axis=1)

            spike_list = np.linspace(0, (signals.shape[1]-(t_max+start)), ntimes, dtype=int)

            msd_matrix = np.zeros((len(spike_list), t_max))
            energy_matrix = np.zeros((len(ds), t_max))
            for i, spike in enumerate(spike_list):
                msd_= msd(spike, start, t_max_, signals)
                msd_matrix[i] = msd_
            for i in range(t_max-1):
                energy_mat_ = p_dist_gauss(msd_matrix[:,i], ds=ds, bandwidth=1)
                energy_matrix[:,i] = energy_mat_
            energy_matrix = energy_matrix[:,:-1]

            if not os.path.exists('energy_landscapes/'+perp+'/'+condition+'/'):
                os.makedirs('energy_landscapes/'+perp+'/'+condition+'/')
            np.save('energy_landscapes/'+perp+'/'+condition+'/energy_matrix.npy', energy_matrix)

# %%
def plot_energy_landscape(x, y, energy_matrix, t_max, ds, 
                          y_max, colour, mode='single', cmap='None'):
    X, Y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    if mode == 'single':
        ax.plot_wireframe(X, Y, energy_matrix, color=colour, lw=0.5)
        ax.set_zlim(1, y_max)
    if mode == 'change':
        max_val = np.max(np.abs(energy_matrix))
        ax.plot_surface(X, Y, energy_matrix, cmap=cmap, lw=0.5, alpha=0.95, vmin=-max_val, vmax=max_val,
                        edgecolor='black')
        ax.set_zlim(-0.1, y_max)
    ax.view_init(5, -40)
    ax.set_xticks(np.arange(0, t_max, 100))
    ax.set_xticklabels(np.arange(0, t_max, 100)*2)
    ax.set_yticks(np.arange(0, np.max(ds), 1))
    ax.set_yticklabels(np.arange(0, np.max(ds), 1))
    ax.set_xlabel('Lag (ms)')
    ax.set_ylabel('Effect (MSD)')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.tick_params(axis='both', which='major', pad=0.2)
    ax.tick_params(axis='both', which='minor', pad=0.2)
    ax.zaxis.labelpad=-0.7

def plot_energy_heatmap(energy_matrix, t_max, ds, vmin, vmax, 
                        cmode='single', cmap='None', x=True):
    fig, ax = plt.subplots(1,1, figsize=(5,1.275))
    # plot the energy matrix where columns are lags and rows are effects and the values are energy
    ax.imshow(energy_matrix, cmap=cmap, aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_ylabel('MSD')
    if x==True:
        ax.set_xticks(np.arange(0, t_max-1, 100))
        ax.set_xticklabels(np.arange(0, t_max-1, 100)*2)
        ax.set_xlabel('Lag (ms)')
    else:
        ax.set_xticklabels([])
        ax.set_xlabel('')
    ax.set_xlim(0, t_max-1)
    ax.set_ylim(0, len(ds)-1)
    ax.set_yticks(np.arange(0, len(ds), np.round(len(ds)/2)))
    ax.set_yticklabels(np.round(ds[::int(np.round(len(ds)/2))],2))
    cbar = plt.colorbar(ax.imshow(energy_matrix, cmap=cmap, aspect='auto', origin='lower', vmin=vmin, vmax=vmax))
    if cmode == 'single':
        cbar.set_label('Energy')
    if cmode == 'change':
        cbar.set_label(r'$\Delta$ Energy')


x = np.arange(start, t_max+(start-1))
y = ds

# %%
energy_matrix_rest = np.zeros((len(ds), t_max-1))
energy_matrix_five = np.zeros((len(ds), t_max-1))
energy_matrix_diff = np.zeros((len(ds), t_max-1))

for perp in perps:
    energy_matrix_rest += np.load('energy_landscapes/'+perp+'/Rest/energy_matrix.npy')
    energy_matrix_five += np.load('energy_landscapes/'+perp+'/5-MeO-DMT/energy_matrix.npy')
    energy_matrix_diff += np.load('energy_landscapes/'+perp+'/5-MeO-DMT/energy_matrix.npy') - np.load('energy_landscapes/'+perp+'/Rest/energy_matrix.npy')

energy_matrix_rest = energy_matrix_rest / len(perps)
energy_matrix_five = energy_matrix_five / len(perps)
energy_matrix_diff = energy_matrix_diff / len(perps)

# create a colour map thats the first half of PRGn and the second half of PRGn
cmaplen = 1000
PRGn = sns.color_palette('PRGn_r', cmaplen)
PR = PRGn[cmaplen//2:]
Gn = PRGn[:cmaplen//2]
import matplotlib as mpl
PR = mpl.colors.ListedColormap(PR)
Gn = mpl.colors.ListedColormap(Gn)
Gn = Gn.reversed()

vmin = 1
vmax = 4.8
plot_energy_heatmap(energy_matrix_rest, t_max, ds, vmin, vmax, 'single', cmap=Gn, x=False)
plot_energy_heatmap(energy_matrix_five, t_max, ds, vmin, vmax, 'single', cmap=PR, x=False)
plot_energy_heatmap(energy_matrix_diff, t_max, ds, -1, 1, 'change', cmap='PRGn_r')

plot_energy_landscape(x,y, energy_matrix_rest, t_max, ds, 5.5, 'g', 'single')
plot_energy_landscape(x,y, energy_matrix_five, t_max, ds, 5.5, 'purple', 'single')
plot_energy_landscape(x,y, energy_matrix_diff, t_max, ds, 5.5, 'r', 'change', cmap='PRGn_r')
    

# %%


# %%
y = ds

for perp in perps:
    energy_matrix_r = np.load('energy_landscapes/'+perp+'/Rest/energy_matrix.npy')
    energy_matrix_f = np.load('energy_landscapes/'+perp+'/5-MeO-DMT/energy_matrix.npy')
    energy_matrix_r_mean = np.mean(energy_matrix_r, axis=1)
    energy_matrix_f_mean = np.mean(energy_matrix_f, axis=1)
    if perp == perps[0]:
        energy_matrix_rest_coll = energy_matrix_r_mean
        energy_matrix_five_coll = energy_matrix_f_mean
    else:
        energy_matrix_rest_coll = np.vstack((energy_matrix_rest_coll, energy_matrix_r_mean))
        energy_matrix_five_coll = np.vstack((energy_matrix_five_coll, energy_matrix_f_mean))

# make into a dataframe to save
df = pd.DataFrame(columns=['MSD', 't', 'p', 'cil', 'ciu', 'bf', 'd', 'dof', 'power', 'p_bh'])

for i in range(len(y)):
    t, p, cil, ciu, d, bf, dof, power = aug_t_test(energy_matrix_five_coll[:,i], energy_matrix_rest_coll[:,i])
    df = df.append({'MSD': y[i], 't': t, 'p': p, 'cil': cil, 'ciu': ciu, 'bf': bf, 'd': d, 'dof': dof, 'power': power}, ignore_index=True)
df['p_bh'] = multi.multipletests(df['p'], method='fdr_bh')[1]
df.to_csv('energy_landscapes/stats.csv')

energy_matrix_rest_mean = np.mean(energy_matrix_rest_coll, axis=0)
energy_matrix_five_mean = np.mean(energy_matrix_five_coll, axis=0)
energy_matrix_rest_std = np.std(energy_matrix_rest_coll, axis=0)/(len(perps)**0.5)
energy_matrix_five_std = np.std(energy_matrix_five_coll, axis=0)/(len(perps)**0.5)

plt.figure(figsize=(4.5, 4.5))
plt.plot(y, energy_matrix_rest_mean, 'g', label='Rest', linestyle='-')
plt.fill_between(y, energy_matrix_rest_mean-energy_matrix_rest_std, energy_matrix_rest_mean+energy_matrix_rest_std, color='g', alpha=0.1)
plt.plot(y, energy_matrix_five_mean, 'purple', label='5-MeO-DMT')
plt.fill_between(y, energy_matrix_five_mean-energy_matrix_five_std, energy_matrix_five_mean+energy_matrix_five_std, color='purple', alpha=0.1)
plt.xlabel('Effect (MSD)')
plt.ylabel('Energy')
#plt.legend(loc='upper left')
#plt.yscale('log', base=2)
sns.despine()
plt.yticks(np.arange(1, np.max(ds), 2))

# if there are significant differences, plot them
for i in range(len(y)):
    if df['p_bh'][i] < 0.05:
        plt.text(y[i], energy_matrix_rest_mean[i]+energy_matrix_rest_std[i], '*', ha='center', va='center', fontsize=20, color='black')
    elif df['p'][i] < 0.1:
        plt.text(y[i], energy_matrix_rest_mean[i]+energy_matrix_rest_std[i], '*', ha='center', va='center', fontsize=20, color='grey')
    

# %%
df = pd.read_csv('energy_landscapes/stats.csv')
df = df.dropna()
T_max = df['T'].idxmax()
t = write_scientific(df['t'][T_max])
p = write_scientific(df['p_bh'][T_max])
b = write_scientific(df['bf'][T_max])
d = write_scientific(df['d'][T_max])
print('MSD max:', np.round(df['MSD'][T_max], 2))
print(f'$T=${t}, $p_{{FDR}}=${p}, $BF_{{10}}=${b}, $d=${d}.')
T_min = df['T'].idxmin()
t = write_scientific(df['t'][T_min])
p = write_scientific(df['p_bh'][T_min])
b = write_scientific(df['bf'][T_min])
d = write_scientific(df['d'][T_min])
print('MSD min:', np.round(df['MSD'][T_min], 2))
print(f'$T=${t}, $p_{{FDR}}=${p}, $BF_{{10}}=${b}, $d=${d}.')

# %%
from scipy.optimize import curve_fit
import pingouin as pg

# exponential
def curve_func(x, slope):
    return np.exp(slope*x)


def fitting(source, target):
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(curve_func, source, target)
    slope = popt
    target_pred = [curve_func(x_, slope) for x_ in source]
    # calculate adjusted R2 = 1 - (SSres/SStot)
    # (adjusted for number of parameters) = 1 - (SSres/(n-p)) / (SStot/(n-1))
    # where SSres = sum of squared residuals, SStot = total sum of squares, n = number of observations, p = number of parameters
    SSres = np.sum(np.square(np.subtract(target, target_pred)))
    SStot = np.sum(np.square(np.subtract(target, np.mean(target))))
    n = len(target)
    p = 1
    R2 = 1 - (SSres/SStot)
    adj_R2 = 1 - ((SSres/(n-p)) / (SStot/(n-1)))
    return slope, adj_R2

y = ds

slopes_df = pd.DataFrame(columns=['perp', 'condition', 'slope', 'adj_R2'])
for perp in perps:
    energy_matrix_r = np.load('energy_landscapes/'+perp+'/Rest/energy_matrix.npy')
    energy_matrix_f = np.load('energy_landscapes/'+perp+'/5-MeO-DMT/energy_matrix.npy')
    energy_matrix_r_mean = np.mean(energy_matrix_r, axis=1)
    energy_matrix_f_mean = np.mean(energy_matrix_f, axis=1)
    # fit a line to the mean energy landscape
    slope_r, adj_R2_r = fitting(y, energy_matrix_r_mean)
    slope_f, adj_R2_f = fitting(y, energy_matrix_f_mean)
    slope_r = slope_r[0]
    slope_f = slope_f[0]
    # store the slopes
    slopes_df = slopes_df.append({'perp': perp, 'condition': 'Rest', 'slope': slope_r, 'adj_R2': adj_R2_r}, ignore_index=True)
    slopes_df = slopes_df.append({'perp': perp, 'condition': '5-MeO-DMT', 'slope': slope_f, 'adj_R2': adj_R2_f}, ignore_index=True)
slopes_df.to_csv('energy_landscapes/landscape_slopes.csv')
slope_stats = pd.DataFrame(columns=['t', 'p', 'cil', 'ciu', 'bf', 'd', 'dof', 'power']
                           , index=['slope', 'adj_R2'])
for param in ['slope', 'adj_R2']:
    rest_ = slopes_df[slopes_df['condition'] == 'Rest'][param]
    five_ = slopes_df[slopes_df['condition'] == '5-MeO-DMT'][param]
    t, p, cil, ciu, d, bf, dof, power = aug_t_test(five_, rest_, paired=True)
    slope_stats.loc[param] = [t, p, cil, ciu, bf, d, dof, power]
print(slope_stats)
slope_stats.to_csv('energy_landscapes/landscape_slope_stats.csv')

# write the stats
T = write_scientific(slope_stats['t']['slope'])
p = write_scientific(slope_stats['p']['slope'])
b = write_scientific(slope_stats['bf']['slope'])
d = write_scientific(slope_stats['d']['slope'])
print(f'$T=${T}, $p_{{FDR}}=${p}, $BF_{{10}}=${b}, $d=${d}.')


# plot simulated curves
slopes_df = pd.read_csv('energy_landscapes/landscape_slopes.csv')
plt.figure(figsize=(6, 5))
for perp in perps:
    for condition in conditions:
        if condition == 'Rest':
            colour = 'g'
            lins = '--'
        else:
            colour = 'purple'
            lins = '-'
        sim_energy_matrix = curve_func(y, slopes_df[(slopes_df['perp'] == perp) & (slopes_df['condition'] == condition)]['slope'].values[0])
        plt.plot(y, sim_energy_matrix, colour, linestyle=lins, alpha=0.5)
plt.xlabel('Effect (MSD)')
plt.ylabel('Energy')
sns.despine()



# %%



