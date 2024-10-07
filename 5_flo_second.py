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

    result = pg.wilcoxon(array1, array2, correction=False)

    w = result['W-val'].values[0]
    p = result['p-val'].values[0]
    rbc = result['RBC'].values[0] # matched pairs rank biserial correlation
    cles = result['CLES'].values[0] # common language effect size

    return w, p, rbc, cles

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

mode_dict = {1: 'Delta', 2: 'Slow', 3: 'Ultra-Slow'}

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


# %%
# look at the mean frequency of each mode
imf_if_df = pd.DataFrame(columns=['perp', 'condition', 'mode', 'IF'])
for perp in perps:
    for condition in conditions:
        IF_df = pd.read_csv('modes/freqs/'+perp+'/'+condition+'/freqs.csv', index_col=0)
        mean_IF = IF_df.mean(axis=1)
        for mode in [1,2,3,4,5,6]:
            imf_if_df = imf_if_df.append({'perp': perp, 'condition': condition, 'mode': mode, 'IF': mean_IF[mode-1]}, ignore_index=True)

# do stats
imf_if_stats_df = pd.DataFrame(columns=['mode', 't', 'p', 'cil', 'ciu', 'd', 'bf', 'dof', 'power'])
for mode in [1,2,3,4,5,6]:
    rest_IF = imf_if_df[(imf_if_df['mode']==mode) & (imf_if_df['condition']=='Rest')]['IF'].values
    five_IF = imf_if_df[(imf_if_df['mode']==mode) & (imf_if_df['condition']=='5-MeO-DMT')]['IF'].values
    t, p, cil, ciu, d, bf, dof, power = aug_t_test(five_IF, rest_IF)
    imf_if_stats_df = imf_if_stats_df.append({'mode': mode, 't': t, 'p': p, 'cil': cil, 'ciu': ciu, 'd': d, 'bf': bf, 'dof': dof, 'power': power}, ignore_index=True)
# correct for multiple comparisons
imf_if_stats_df['p_bf'] = multi.multipletests(imf_if_stats_df['p'], method='bonferroni')[1]


fig, ax = plt.subplots(1, 1, figsize=(6.5, 5))
sns.boxplot(data=imf_if_df, x='mode', y='IF', hue='condition', fill=False,
            palette={'Rest': 'green', '5-MeO-DMT': 'purple'}, ax=ax, dodge=True)
ax.set_yscale('log')
ax.set_xlabel('IMF')
ax.set_ylabel(r'$\overline{IF}$ (Hz)')
x_min, x_max = ax.get_xlim()
ax.legend(loc='lower left')
sns.despine()

for band in band_range_dict:
    # want a line in y axis not x so its not axvline its 
    plt.plot([x_min, x_max], [band_range_dict[band][0], band_range_dict[band][0]], color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.plot([x_min, x_max], [band_range_dict[band][1], band_range_dict[band][1]], color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    pos = (band_range_dict[band][0]+band_range_dict[band][1])/2
    plt.text(x_max, pos, band_symbol_dict[band], color='grey', ha='center')

for mode in [1,2,3,4,5,6]:
    max_y = imf_if_df[imf_if_df['mode']==mode]['IF'].max()
    p = imf_if_stats_df[imf_if_stats_df['mode']==mode]['p_bf'].values[0]
    if p < 0.001:
        plt.text(mode-1, max_y, '***', ha='center', va='bottom', fontsize=20)
    elif p < 0.01:
        plt.text(mode-1, max_y, '**', ha='center', va='bottom', fontsize=20)
    elif p < 0.05:
        plt.text(mode-1, max_y, '*', ha='center', va='bottom', fontsize=20)

# shade in grey the modes that we use (3,4,5)
plt.axvspan(1.5, 4.5, color='grey', alpha=0.1)

# print the mean IF for each mode for each condition
for mode in [1,2,3,4,5,6]:
    # rest_IF = imf_if_df[(imf_if_df['mode']==mode) & (imf_if_df['condition']=='Rest')]['IF'].values
    # five_IF = imf_if_df[(imf_if_df['mode']==mode) & (imf_if_df['condition']=='5-MeO-DMT')]['IF'].values
    # print(f'Mode {mode}: Rest {np.mean(rest_IF):.2f} 5-MeO-DMT {np.mean(five_IF):.3f}')
    IF = imf_if_df[imf_if_df['mode']==mode]['IF'].values
    # print mode and mean IF $\pm$ standard error of the mean
    print(f'{np.mean(IF):.3f}', r'$\pm$', f'{np.std(IF)/np.sqrt(len(perps)):.3f}')

# then also just plot modes 3,4,5
imf_if_df_ = imf_if_df[imf_if_df['mode'].isin([3,4,5])]
fig, ax = plt.subplots(1, 1, figsize=(3, 5))
sns.boxplot(data=imf_if_df_, x='mode', y='IF', hue='condition', fill=False, width=0.5,
            palette={'Rest': 'green', '5-MeO-DMT': 'purple'}, ax=ax, dodge=True)
ax.set_xticklabels(['1', '2', '3'])
ax.set_xlabel('Mode')
ax.set_ylabel(r'$\overline{IF}$ (Hz)')
x_min, x_max = ax.get_xlim()
# make custom legend with green for rest and purple for '5-MeO'
custom_lines = [plt.Line2D([0], [0], color='green', lw=2),
                plt.Line2D([0], [0], color='purple', lw=2)]
ax.legend(custom_lines, ['Rest', '5-MeO'], loc='upper right')


sns.despine()


imf_if_stats_df

# %%
# plot a histogram of the angles of the flow field

bin_res = 50
 
run_angle = False
if run_angle:
    for mode in modes:
        rest_hist = np.zeros((bin_res, len(perps)))
        five_hist = np.zeros((bin_res, len(perps)))
        for perp in perps:
            for condition in conditions:
                u = np.load(f'flow_fields/{perp}/{condition}/mode_{mode}/u.npy')
                v = np.load(f'flow_fields/{perp}/{condition}/mode_{mode}/v.npy')
                angles = np.arctan2(v, u)
                angles = angles.flatten()
                angles = angles[~np.isnan(angles)]
                # make a histogram between 0 and 2pi
                hist = np.histogram(angles, bins=bin_res, range=(-np.pi, np.pi), density=True)
                hist = hist[0]
                if condition == 'Rest':
                    rest_hist[:, perps.index(perp)] = hist
                else:
                    five_hist[:, perps.index(perp)] = hist
        # save
        np.save(f'flow_fields/dir_hist_mode_{mode}_rest.npy', rest_hist)
        np.save(f'flow_fields/dir_hist_mode_{mode}_5.npy', five_hist)

run_angle_stats = True
if run_angle_stats:

    angle_stats_df = pd.DataFrame(columns=['mode', 'angle', 't', 'p', 'cil', 'ciu', 'd', 'bf', 'dof', 'power'])

    direc_T = np.zeros((bin_res, len(modes)))
    direc_P = np.zeros((bin_res, len(modes)))
    for mode in modes:
        rest_hist = np.load(f'flow_fields/dir_hist_mode_{mode}_rest.npy')
        five_hist = np.load(f'flow_fields/dir_hist_mode_{mode}_5.npy')
        for i in range(bin_res):
            t, p, cil, ciu, d, bf, dof, power = aug_t_test(five_hist[i, :], rest_hist[i, :])
            direc_T[i, mode-3] = t
            direc_P[i, mode-3] = p
            angle = np.linspace(-np.pi, np.pi, bin_res)[i]
            angle_stats_df = angle_stats_df.append({'mode': mode, 'angle': angle, 't': t, 'p': p, 'cil': cil, 'ciu': ciu, 'd': d, 'bf': bf, 'dof': dof, 'power': power}, ignore_index=True)
    
    direc_P_bh = multi.multipletests(direc_P.flatten(), alpha=0.05, method='fdr_bh')[1]
    direc_P_bh = direc_P_bh.reshape((bin_res, len(modes)))
    np.save('flow_fields/direc_T.npy', direc_T)
    np.save('flow_fields/direc_P_bh.npy', direc_P_bh)

    angle_stats_df['p_bh'] = multi.multipletests(angle_stats_df['p'], method='fdr_bh')[1]
    angle_stats_df.to_csv('flow_fields/angle_stats.csv')

else:
    direc_T = np.load('flow_fields/direc_T.npy')
    direc_P_bh = np.load('flow_fields/direc_P_bh.npy')
    
for mode in modes:
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(polar=True), figsize=(10, 10))
    # rest_hist = np.load(f'flow_fields/dir_hist_mode_{mode}_rest.npy')
    # five_hist = np.load(f'flow_fields/dir_hist_mode_{mode}_5.npy')
    # rest = np.mean(rest_hist, axis=1)
    # five = np.mean(five_hist, axis=1)
    # ax.bar(np.linspace(-np.pi, np.pi, bin_res), rest, width=2*np.pi/bin_res, alpha=1, color='green', edgecolor='black')
    # ax.bar(np.linspace(-np.pi, np.pi, bin_res), five, width=2*np.pi/bin_res, alpha=1, color='purple', edgecolor='black')
    t = direc_T[:, mode-3]
    p = direc_P_bh[:, mode-3]
    vmin = -11; vmax = 11
    cmap = 'PRGn_r'
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    p_thresh = 0.05
    for i in range(bin_res):
        if direc_P_bh[i, mode-3] < p_thresh:
            ax.bar(np.linspace(-np.pi, np.pi, bin_res)[i], t[i], width=2*np.pi/bin_res, alpha=1, color=sm.to_rgba(t[i]), edgecolor='black')
        else:
            ax.bar(np.linspace(-np.pi, np.pi, bin_res)[i], t[i], width=2*np.pi/bin_res, alpha=1, color=sm.to_rgba(t[i]))
        # ax.bar(np.linspace(-np.pi, np.pi, bin_res)[i], t[i], width=2*np.pi/bin_res, alpha=1, color=sm.to_rgba(t[i]))
        # if direc_P_bh[i, mode-3] < 0.05:
        #     if t[i] < 0:
        #         hwidth = 0.1
        #         hlength = 0.1
        #         c = 'green'
        #     else:
        #         hwidth = 0.01
        #         hlength = 0.1
        #         c = 'purple'
        #     c='k'
        #     ax.arrow(np.linspace(-np.pi, np.pi, bin_res)[i], 0, 0, t[i], head_width=hwidth, head_length=hlength, fc=c, ec=c)
    if mode == 5:
        fs = 20
    else:
        fs = 35
    ax.text(0, -12, mode_dict[mode-2], ha='center', va='center', fontsize=fs)
    #ax.text(0, -12, f'Mode {mode-2}', ha='center', va='center', fontsize=fs)
    ax.set_ylim(-12, 6)
    ax.yaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(alpha=0.2)
    #ax.set_xticklabels(['', '', r'Anterior', '', '', '', r'Posterior'])
    ax.set_xticklabels(['', '', '', '', '', '', ''])
    plt.show()

# %%
# print the part of the angle_stats_df for the forward and backward angles for mode 5
# posterior to anterior  = -1.60285339468867
# anterior to posterior = 1.6028533946886698

angle_stats_df = pd.read_csv('flow_fields/angle_stats.csv', index_col=0)

# write paper text
for mode in [3,4,5]:
    print(f'Mode {mode-2}:')
    for angl in [-1.60285339468867, 1.6028533946886698]:
        t = write_scientific(angle_stats_df[(angle_stats_df['mode']==mode) & (angle_stats_df['angle']==angl)]['t'].values[0])
        p = write_scientific(angle_stats_df[(angle_stats_df['mode']==mode) & (angle_stats_df['angle']==angl)]['p_bh'].values[0])
        b = write_scientific(angle_stats_df[(angle_stats_df['mode']==mode) & (angle_stats_df['angle']==angl)]['bf'].values[0])
        d = write_scientific(angle_stats_df[(angle_stats_df['mode']==mode) & (angle_stats_df['angle']==angl)]['d'].values[0])
        if angl < 0:
            direction = 'Anterior travel'
        else:
            direction = 'Posterior travel'
        print(f'{direction}: $T=${t}, $p_{{FDR}}=${p}, $BF_{{10}}=${b}, $d=${d};')
        # print(f'{direction}: $p_{{FDR}}=${p};')
        # print(f'{direction}: $d=${t}.')


# %%
vmin=-11; vmax=11
fig, ax = plt.subplots(figsize=(0.15, 2))
import matplotlib as mpl
cmap = mpl.cm.PRGn_r
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')
plt.xticks(rotation=90)
cb1.set_ticks([vmin, 0, vmax])
cb1.set_label(r'T', rotation=0)
plt.show()


# %%
run_order = False
if run_order:

    order_df = pd.DataFrame(columns=['perp', 'condition', 'mode', 'phi', 'H', 'speed'])
    for perp in perps:
        for condition in conditions:
            for mode in modes:
                H = np.load(f'flow_fields/{perp}/{condition}/mode_{mode}/H.npy')
                H = np.mean(H)
                phi = np.load(f'flow_fields/{perp}/{condition}/mode_{mode}/Phi.npy')
                phi = np.mean(phi)
                u = np.load(f'flow_fields/{perp}/{condition}/mode_{mode}/u.npy')
                v = np.load(f'flow_fields/{perp}/{condition}/mode_{mode}/v.npy')
                speed = np.sqrt(u**2 + v**2)
                speed = speed[~np.isnan(speed)]
                speed = np.mean(speed)
                order_df = order_df.append({'perp': perp, 'condition': condition, 'mode': mode, 
                                                'phi': phi, 'H': H, 'speed': speed}, ignore_index=True)
    # save
    order_df.to_csv('flow_fields/order_df.csv')
else:
    order_df = pd.read_csv('flow_fields/order_df.csv', index_col=0)

# stats
order_stats_df = pd.DataFrame(columns=['mode', 'measure', 't', 'p', 'cil', 'ciu', 'd', 'bf', 'dof', 'power'])
for measure in ['phi', 'H', 'speed']:
    for mode in modes:
        rest = order_df[(order_df['mode']==mode) & (order_df['condition']=='Rest')][measure].values
        five = order_df[(order_df['mode']==mode) & (order_df['condition']=='5-MeO-DMT')][measure].values
        t, p, cil, ciu, d, bf, dof, power = aug_t_test(five, rest)
        order_stats_df = order_stats_df.append({'mode': mode, 'measure': measure, 't': t, 'p': p, 'cil': cil, 'ciu': ciu, 'd': d, 'bf': bf, 'dof': dof, 'power': power}, ignore_index=True)
# correct for multiple comparisons
order_stats_df['p_bh'] = multi.multipletests(order_stats_df['p'], method='fdr_bh')[1]
order_stats_df.to_csv('flow_fields/order_stats_df.csv')

# plot
for measure in ['phi', 'H', 'speed']:
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    sns.violinplot(data=order_df, x='mode', y=measure, hue='condition', ax=ax,
               split=True, linewidth=1.3, palette={"Rest": "green", "5-MeO-DMT": "purple"}, 
               saturation=1, gap=0.1, alpha=1, cut=0, fill=False)
    ax.get_legend().remove()
    if measure == 'phi':
        ax.set_ylabel('Field Velocity ($\overline{\Phi}$)')
    elif measure == 'H':
        ax.set_ylabel('Field Hereogeneity ($H$)')
    else:
        ax.set_ylabel('Average Speed ($cm/s$)')
    ax.set_xticklabels([str(mode_dict[m-2]) for m in modes])
    # ax.set_xlabel('Mode')
    ax.set_xlabel('')

    for j, mode in enumerate(modes):
        max_val = order_df[order_df['mode']==mode][measure].max()
        p = order_stats_df[(order_stats_df['mode']==mode) & (order_stats_df['measure']==measure)]['p_bh'].values[0]
        if p < 0.001:
            plt.text(j, max_val, '***', ha='center', va='bottom', fontsize=20)
        elif p < 0.01:
            plt.text(j, max_val, '**', ha='center', va='bottom', fontsize=20)
        elif p < 0.05:
            plt.text(j, max_val, '*', ha='center', va='bottom', fontsize=20)

    sns.despine()

# write paper text
for measure in ['phi', 'H', 'speed']:
    print(f'{measure}:')
    for mode in modes:
        # t = write_scientific(order_stats_df[(order_stats_df['mode']==mode) & (order_stats_df['measure']==measure)]['t'].values[0])
        p = write_scientific(order_stats_df[(order_stats_df['mode']==mode) & (order_stats_df['measure']==measure)]['p_bh'].values[0])
        # b = write_scientific(order_stats_df[(order_stats_df['mode']==mode) & (order_stats_df['measure']==measure)]['bf'].values[0])
        # d = write_scientific(order_stats_df[(order_stats_df['mode']==mode) & (order_stats_df['measure']==measure)]['d'].values[0])
        # print(f'Mode {mode-2}: $T=${t}, $p_{{FDR}}=${p}, $BF_{{10}}=${b}, $d=${d}.')
        # print(f'Mode {mode-2}: $p_{{FDR}}=${p}.')
        print(f'{mode_dict[mode-2]}: $p_{{FDR}}=${p};')
        

# %%
# correlate changes in H with changes in Phi
order_change_df = pd.DataFrame(columns=['perp', 'mode', 'H', 'Phi'])

for perp in perps:
    for mode in modes:
        rest_H = np.mean(np.load(f'flow_fields/{perp}/Rest/mode_{mode}/H.npy'))
        five_H = np.mean(np.load(f'flow_fields/{perp}/5-MeO-DMT/mode_{mode}/H.npy'))
        rest_Phi = np.mean(np.load(f'flow_fields/{perp}/Rest/mode_{mode}/phi.npy'))
        five_Phi = np.mean(np.load(f'flow_fields/{perp}/5-MeO-DMT/mode_{mode}/phi.npy'))
        H_change = five_H - rest_H
        Phi_change = five_Phi - rest_Phi
        order_change_df = order_change_df.append({'perp': perp, 'mode': mode, 'H': H_change, 'Phi': Phi_change}, ignore_index=True)

# plot
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(data=order_change_df, x='H', y='Phi', hue='mode', ax=ax, palette='copper_r', s=100, marker='o')
plt.xlabel(r'$\Delta H$')
plt.ylabel(r'$\Delta \overline{v_\phi}$')
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.axvline(0, color='black', linestyle='--', alpha=0.5)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=[f'Mode {m-2}' for m in modes])
sns.despine()

# %%
# plot the mean and std error kuramoto order parameter

# open an example R
R = np.load(f'flow_fields/{perps[0]}/Rest/mode_3/R.npy')
n_t = len(R)

for mode in modes:
    rest = np.zeros((n_t, len(perps)))
    five = np.zeros((n_t, len(perps)))
    for i, perp in enumerate(perps):
        rest_r = np.load(f'flow_fields/{perp}/Rest/mode_{mode}/R.npy')
        five_r = np.load(f'flow_fields/{perp}/5-MeO-DMT/mode_{mode}/R.npy')
        rest[:, i] = rest_r
        five[:, i] = five_r
    rest_mean = np.mean(rest, axis=1); rest_se = np.std(rest, axis=1) / np.sqrt(len(perps))
    five_mean = np.mean(five, axis=1); five_se = np.std(five, axis=1) / np.sqrt(len(perps))

    fig, ax = plt.subplots(1, 1, figsize=(15, 1.5))
    ax.plot(np.linspace(0, 1, n_t), rest_mean, color='green', label='Rest', linewidth=.5)
    ax.fill_between(np.linspace(0, 1, n_t), rest_mean - rest_se, rest_mean + rest_se, color='green', alpha=0.2)
    ax.plot(np.linspace(0, 1, n_t), five_mean, color='purple', label='5-MeO-DMT', linewidth=.5)
    ax.fill_between(np.linspace(0, 1, n_t), five_mean - five_se, five_mean + five_se, color='purple', alpha=0.2)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels([0, 15, 30, 45, 60])
    plt.xlabel('Time (s)')
    plt.ylabel(r'$R$')
    if mode != 5:
        # remove x-axis 
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_xlabel('')
    sns.despine()


# %%
stack_coh = False

if stack_coh:
    for i, mode in enumerate(modes):
        for j, perp in enumerate(perps):
            for k, condition in enumerate(conditions):
                coherence_df = pd.read_csv(f'flow_fields/{perp}/{condition}/mode_{mode}/coherence_df.csv', index_col=0)
                if i == 0 and j == 0 and k == 0:
                    coh_df = coherence_df.copy()
                else:
                    coh_df = pd.concat([coh_df, coherence_df], axis=0)
    coh_df.to_csv('flow_fields/coherence_df.csv')
else:
    coh_df = pd.read_csv('flow_fields/coherence_df.csv', index_col=0)


local_radi = np.arange(2, 10, 1)
res = 32
nas_in = 0.35
d0 = 0.8*nas_in
# convert x units to cm
# 1 pixel in cm is (d0*100)/res
local_radi_cm = local_radi * (d0*100)/res

measures = ['Global Coherence', 'Global Metastability']
for radi in local_radi:
    measures.append(f'Local Coherence {radi}')
    measures.append(f'Local Metastability {radi}')
    measures.append(f'TP-V Coherence r{radi}')
    measures.append(f'TP-V Metastability r{radi}')
measures_s = ['Global Coherence', 'Global Metastability']
for radi in local_radi:
    measures_s.append(f'Local Coherence {radi}')
    measures_s.append(f'Local Metastability {radi}')

coh_stats_df = pd.DataFrame(columns=['measure', 'mode', 't', 'p', 'cil', 'ciu', 'd', 'bf', 'dof', 'power'])

for measure in measures_s:
    for mode in modes:
        rest = coh_df[(coh_df['condition'] == 'Rest') & (coh_df['mode'] == mode)][measure].values
        five = coh_df[(coh_df['condition'] == '5-MeO-DMT') & (coh_df['mode'] == mode)][measure].values
        t, p, cil, ciu, d, bf, dof, power = aug_t_test(five, rest)
        coh_stats_df = coh_stats_df.append({'measure': measure, 'mode': mode, 't': t, 'p': p, 'cil': cil, 'ciu': ciu, 'd': d, 'bf': bf, 'dof': dof, 'power': power}, ignore_index=True)
coh_stats_df['p_bh'] = multi.multipletests(coh_stats_df['p'], method='fdr_bh')[1]

coh_stats_df.to_csv('flow_fields/coh_stats_df.csv')

# make a plot for each measure
for measure in ['Global Coherence', 'Global Metastability']:
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    sns.violinplot(x='mode', y=measure, hue='condition', data=coh_df, ax=ax,
                   split=True, linewidth=1.3, palette={"Rest": "green", "5-MeO-DMT": "purple"}, saturation=1, gap=0.1, alpha=1, cut=0, fill=False)
    ax.get_legend().remove()
    # plt.xlabel('Mode')
    # ax.set_xticklabels([str(m-2) for m in modes])
    ax.set_xticklabels([str(mode_dict[m-2]) for m in modes])
    ax.set_xlabel('')
    plt.ylabel(measure)
    for j, mode in enumerate(modes):
        max_val = coh_df[coh_df['mode'] == mode][measure].max()
        p= coh_stats_df[(coh_stats_df['mode'] == mode) & (coh_stats_df['measure'] == measure)]['p_bh'].values[0]
        if p < 0.001:
            ax.text(j-0.15, max_val, '***', fontsize=20)
        elif p < 0.01:
            ax.text(j-0.12, max_val, '**', fontsize=20)
        elif p < 0.05:
            ax.text(j-0.06, max_val, '*', fontsize=20)
    if measure == 'Global Coherence':
        plt.ylabel(r'Global Coherence ($C$)')
    elif measure == 'Global Metastability':
        plt.ylabel(r'Global Metastability ($K$)')
    sns.despine()

for measure in ['Local Coherence', 'Local Metastability']:
    rest_means = np.zeros((len(local_radi), len(modes)))
    rest_se = np.zeros((len(local_radi), len(modes)))
    five_means = np.zeros((len(local_radi), len(modes)))
    five_se = np.zeros((len(local_radi), len(modes)))
    for i, radi in enumerate(local_radi):
        for j, mode in enumerate(modes):
            radi_ = radi
            if 'TP-V' in measure:
                radi_ = f'r{radi}'
            rest = coh_df[(coh_df['condition'] == 'Rest') & (coh_df['mode'] == mode)][f'{measure} {radi_}'].values
            five = coh_df[(coh_df['condition'] == '5-MeO-DMT') & (coh_df['mode'] == mode)][f'{measure} {radi_}'].values
            rest_means[i, j] = np.mean(rest)
            rest_se[i, j] = np.std(rest) / np.sqrt(len(rest))
            five_means[i, j] = np.mean(five)
            five_se[i, j] = np.std(five) / np.sqrt(len(five))
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    for j, mode in enumerate(modes):
        if mode == 3:
            marker = ':'
        if mode == 4:
            marker = '--'
        if mode == 5:
            marker = '-'
        ax.plot(local_radi_cm, rest_means[:, j], color='green', label='Rest', linestyle=marker)
        ax.fill_between(local_radi_cm, rest_means[:, j] - rest_se[:, j], rest_means[:, j] + rest_se[:, j], color='green', alpha=0.1, linestyle=marker)
        ax.plot(local_radi_cm, five_means[:, j], color='purple', label='5-MeO-DMT', linestyle=marker)
        ax.fill_between(local_radi_cm, five_means[:, j] - five_se[:, j], five_means[:, j] + five_se[:, j], color='purple', alpha=0.1, linestyle=marker)
    plt.xlabel('Radius (cm)')
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='black', linestyle=':'),
                    Line2D([0], [0], color='black', linestyle='--'),
                    Line2D([0], [0], color='black', linestyle='-')]
    ax.legend(custom_lines, [f'{mode_dict[m-2]}' for m in modes], loc='upper right', frameon=False)
    if measure == 'Local Coherence':
        plt.ylabel(r'Local Coherence ($\overline{C}_{l}$)')
    elif measure == 'Local Metastability':
        plt.ylabel(r'Local Metastability ($\overline{K}_{l}$)')
    elif measure == 'TP-V Coherence':
        plt.ylabel('TP-V Local Coherence ($\sigma^2_{C_{l}}$)')
    elif measure == 'TP-V Metastability':
        plt.ylabel('TP-V Local Metastability ($\sigma^2_{K_{l}}$)')
    else:
        plt.ylabel(measure)
    sns.despine()    

# print paper text
for measure in ['Global Coherence', 'Global Metastability']:
    print(f'{measure}:')
    for mode in modes:
        # t = write_scientific(coh_stats_df[(coh_stats_df['measure'] == measure) & (coh_stats_df['mode'] == mode)]['t'].values[0])
        p = write_scientific(coh_stats_df[(coh_stats_df['measure'] == measure) & (coh_stats_df['mode'] == mode)]['p_bh'].values[0])
        # b = write_scientific(coh_stats_df[(coh_stats_df['measure'] == measure) & (coh_stats_df['mode'] == mode)]['bf'].values[0])
        # d = write_scientific(coh_stats_df[(coh_stats_df['measure'] == measure) & (coh_stats_df['mode'] == mode)]['d'].values[0])
        print(f'{mode_dict[mode-2]}: $p_{{FDR}}=${p};')

# find which radius has the max t-value for the measure
for measure in ['Local Coherence', 'Local Metastability']:
    print(f'{measure}:')
    for mode in modes:
        max_T = 0
        for radi in local_radi:
            t = coh_stats_df[(coh_stats_df['measure'] == f'{measure} {str(radi)}') & (coh_stats_df['mode'] == mode)]['t'].values[0]
            if abs(t) > abs(max_T):
                max_T = t
                max_radius = radi
        p = write_scientific(coh_stats_df[(coh_stats_df['measure'] == f'{measure} {str(max_radius)}') & (coh_stats_df['mode'] == mode)]['p_bh'].values[0])
        print(f'{mode_dict[mode-2]}: $p_{{FDR}}=${p};')

# %%
# make a plot with two subplots for the TP-V measures


fig, ax = plt.subplots(1, 2, figsize=(8, 4))

rest_means_met = np.zeros((len(local_radi), len(modes)))
rest_se_met = rest_means_met.copy(); five_means_met = rest_means_met.copy(); five_se_met = rest_means_met.copy()
rest_means_coh = rest_means_met.copy(); rest_se_coh = rest_means_met.copy(); five_means_coh = rest_means_met.copy(); five_se_coh = rest_means_met.copy()

for i, radi in enumerate(local_radi):
    for j, mode in enumerate(modes):
        radi_ = radi
        rest_met = coh_df[(coh_df['condition'] == 'Rest') & (coh_df['mode'] == mode)][f'TP-V Metastability r{radi}'].values
        five_met = coh_df[(coh_df['condition'] == '5-MeO-DMT') & (coh_df['mode'] == mode)][f'TP-V Metastability r{radi}'].values
        rest_means_met[i, j] = np.mean(rest_met)
        rest_se_met[i, j] = np.std(rest_met) / np.sqrt(len(rest_met))
        five_means_met[i, j] = np.mean(five_met)
        five_se_met[i, j] = np.std(five_met) / np.sqrt(len(five_met))
        rest_coh = coh_df[(coh_df['condition'] == 'Rest') & (coh_df['mode'] == mode)][f'TP-V Coherence r{radi}'].values
        five_coh = coh_df[(coh_df['condition'] == '5-MeO-DMT') & (coh_df['mode'] == mode)][f'TP-V Coherence r{radi}'].values
        rest_means_coh[i, j] = np.mean(rest_coh)
        rest_se_coh[i, j] = np.std(rest_coh) / np.sqrt(len(rest_coh))
        five_means_coh[i, j] = np.mean(five_coh)
        five_se_coh[i, j] = np.std(five_coh) / np.sqrt(len(five_coh))

for j, mode in enumerate(modes):
    if mode == 3:
        marker = ':'
    if mode == 4:
        marker = '--'
    if mode == 5:
        marker = '-'
    ax[1].plot(local_radi_cm, rest_means_met[:, j], color='green', label='Rest', linestyle=marker)
    ax[1].fill_between(local_radi_cm, rest_means_met[:, j] - rest_se_met[:, j], rest_means_met[:, j] + rest_se_met[:, j], color='green', alpha=0.1, linestyle=marker)
    ax[1].plot(local_radi_cm, five_means_met[:, j], color='purple', label='5-MeO-DMT', linestyle=marker)
    ax[1].fill_between(local_radi_cm, five_means_met[:, j] - five_se_met[:, j], five_means_met[:, j] + five_se_met[:, j], color='purple', alpha=0.1, linestyle=marker)
    ax[0].plot(local_radi_cm, rest_means_coh[:, j], color='green', label='Rest', linestyle=marker)
    ax[0].fill_between(local_radi_cm, rest_means_coh[:, j] - rest_se_coh[:, j], rest_means_coh[:, j] + rest_se_coh[:, j], color='green', alpha=0.1, linestyle=marker)
    ax[0].plot(local_radi_cm, five_means_coh[:, j], color='purple', label='5-MeO-DMT', linestyle=marker)
    ax[0].fill_between(local_radi_cm, five_means_coh[:, j] - five_se_coh[:, j], five_means_coh[:, j] + five_se_coh[:, j], color='purple', alpha=0.1, linestyle=marker)

ax[0].set_xlabel('Radius (cm)')
ax[1].set_xlabel('Radius (cm)')
ax[0].set_ylabel('TP-V Coherence')
ax[1].set_ylabel('TP-V Metastability')
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='black', linestyle=':'),
                Line2D([0], [0], color='black', linestyle='--'),
                Line2D([0], [0], color='black', linestyle='-')]
ax[0].legend(custom_lines, [f'{mode_dict[m-2]}' for m in modes], loc='upper right', frameon=False)
sns.despine()
plt.tight_layout()

ax[0].text(-0.1, 1.1, 'a', transform=ax[0].transAxes, fontsize=18, fontweight='bold', va='top', ha='right')
ax[1].text(-0.1, 1.1, 'b', transform=ax[1].transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

# save
fig.savefig('MS1_Figs/Supp_Fig3.png', dpi=500)


# %%
local_radi_cm

# %%
from flo_library import flux_props

flux_props_df = pd.DataFrame(columns=['perp', 'condition', 'mode', 
                                      'n_sources', 'n_sinks', 
                                      'mean_area_sources', 'mean_area_sinks', 
                                      'mean_strength_sources', 'mean_strength_sinks', 
                                      'viscosity',
                                      'ss_a_asym', 'ss_f_asym'])
for mode in modes:
    for perp in perps:
        for condition in conditions:
            singularities_df = pd.read_csv(f'flow_fields/{perp}/{condition}/mode_{mode}/singularities_df.csv')
            n_sources, n_sinks, mean_area_sources, mean_area_sinks, mean_strength_sources, mean_strength_sinks, viscosity, ss_a_asym, ss_f_asym = flux_props(singularities_df)
            flux_props_df = flux_props_df.append({'perp': perp, 'condition': condition, 'mode': mode, 
                                            'n_sources': n_sources, 'n_sinks': n_sinks, 
                                            'mean_area_sources': mean_area_sources, 'mean_area_sinks': mean_area_sinks, 
                                            'mean_strength_sources': mean_strength_sources, 'mean_strength_sinks': mean_strength_sinks,
                                            'viscosity': viscosity, 
                                            'ss_a_asym': ss_a_asym, 'ss_f_asym': ss_f_asym}, ignore_index=True)
            
# save
flux_props_df.to_csv('flow_fields/flux_props.csv', index=False)

# do stats
flux_props_stats_df = pd.DataFrame(columns=['mode', 'measure', 't', 'p', 'cil', 'ciu', 'd', 'bf', 'dof', 'power'])

for mode in modes:
    for measure in ['n_sources', 'n_sinks', 'mean_area_sources', 'mean_area_sinks', 'mean_strength_sources', 'mean_strength_sinks', 'viscosity', 'ss_a_asym', 'ss_f_asym']:
        rest = (flux_props_df[(flux_props_df['condition'] == 'Rest') & (flux_props_df['mode'] == mode)][measure].values).astype(float)
        five = (flux_props_df[(flux_props_df['condition'] == '5-MeO-DMT') & (flux_props_df['mode'] == mode)][measure].values).astype(float)
        t, p, cil, ciu, d, bf, dof, power = aug_t_test(five, rest)
        flux_props_stats_df = flux_props_stats_df.append({'mode': mode, 'measure': measure, 't': t, 'p': p, 'cil': cil, 'ciu': ciu, 'd': d, 'bf': bf, 'dof': dof, 'power': power}, ignore_index=True)

# correct p-values
flux_props_stats_df['p_bh'] = multi.multipletests(flux_props_stats_df['p'], method='fdr_bh')[1]

# save
flux_props_stats_df.to_csv('flow_fields/flux_props_stats_df.csv')

flux_props_stats_df

# %%
# make a plot with 2 rows and 3 columns
fig, ax = plt.subplots(2, 3, figsize=(15, 9))

for ni, measure in enumerate(['n_sources', 'mean_area_sources', 'mean_strength_sources',
                              'n_sinks', 'mean_area_sinks', 'mean_strength_sinks']):
    i = ni // 3
    j = ni % 3
    sns.violinplot(x='mode', y=measure, hue='condition', data=flux_props_df, ax=ax[i, j],
                   split=True, linewidth=1.3, palette={"Rest": "green", "5-MeO-DMT": "purple"}, saturation=1, gap=0.1, alpha=1, cut=0, fill=False)
    plt.xlabel('')
    ax[i, j].get_legend().remove()
    ax[i, j].set_xticklabels([str(mode_dict[m-2]) for m in modes])
    ax[i, j].set_xlabel('')
    ax[i, j].set_ylabel(measure)
    for nm, mode in enumerate(modes):
        max_val = flux_props_df[flux_props_df['mode'] == mode][measure].max()
        p = flux_props_stats_df[(flux_props_stats_df['mode'] == mode) & (flux_props_stats_df['measure'] == measure)]['p_bh'].values[0]
        if p < 0.001:
            ax[i, j].text(nm, max_val, '***', ha='center', va='bottom', fontsize=20)
        elif p < 0.01:
            ax[i, j].text(nm, max_val, '**', ha='center', va='bottom', fontsize=20)
        elif p < 0.05:
            ax[i, j].text(nm, max_val, '*', ha='center', va='bottom', fontsize=20)
    sns.despine()

    if measure == 'n_sources':
        ax[i, j].set_ylabel(r'Number of Sources')
    elif measure == 'n_sinks':
        ax[i, j].set_ylabel(r'Number of Sinks')
    elif measure == 'mean_area_sources':
        ax[i, j].set_ylabel(r'Mean Source Area $(cm^2)$')
    elif measure == 'mean_area_sinks':
        ax[i, j].set_ylabel(r'Mean Sink Area $(cm^2)$')
    elif measure == 'mean_strength_sources':
        ax[i, j].set_ylabel(r'Mean Source Strength ($s^{-1}$)')
    elif measure == 'mean_strength_sinks':
        ax[i, j].set_ylabel(r'Mean Sink Strength ($s^{-1}$)')

    # add labels
    ax[i, j].text(-0.1, 1.1, chr(97+ni), transform=ax[i, j].transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

plt.tight_layout()

# save
fig.savefig('MS1_Figs/Supp_Fig4.png', dpi=500)

# %%
for measure in ['viscosity']:
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    sns.violinplot(x='mode', y=measure, hue='condition', data=flux_props_df, ax=ax,
                   split=True, linewidth=1.3, palette={"Rest": "green", "5-MeO-DMT": "purple"}, saturation=1, gap=0.1, alpha=1, cut=0, fill=False)
    # handles, labels = ax.get_legend_handles_labels(); ax.legend(handles=handles[2:], labels=labels[2:])
    ax.get_legend().remove()
    plt.xlabel('')
    ax.set_xticklabels([str(mode_dict[m-2]) for m in modes])
    plt.ylabel(measure)

    for j, mode in enumerate(modes):
        max_val = flux_props_df[flux_props_df['mode'] == mode][measure].max()
        p= flux_props_stats_df[(flux_props_stats_df['mode'] == mode) & (flux_props_stats_df['measure'] == measure)]['p_bh'].values[0]
        if p < 0.001:
            ax.text(j-0.12, max_val, '***', fontsize=20)
        elif p < 0.01:
            ax.text(j-0.09, max_val, '**', fontsize=20)
        elif p < 0.05:
            ax.text(j-0.06, max_val, '*', fontsize=20)

    if measure == 'n_sources':
        plt.ylabel('Number of Sources')
    elif measure == 'n_sinks':
        plt.ylabel('Number of Sinks')
    elif measure == 'mean_area_sources':
        plt.ylabel(r'Source Area ($cm^2$)')
    elif measure == 'mean_area_sinks':
        plt.ylabel(r'Sink Area ($cm^2$)')
    elif measure == 'mean_strength_sources':
        plt.ylabel('Source Strength ($s^{-1}$)')
    elif measure == 'mean_strength_sinks':
        plt.ylabel('Sink Strength ($s^{-1}$)')
    elif measure == 'viscosity':
        plt.ylabel('Viscosity ($cm^{2}s^{-1}$)')

    else:
        plt.ylabel(measure)

    sns.despine()

# %%
# write scientific for sources and sinks area and strength

for measure in ['mean_area_sources', 'mean_area_sinks', 'mean_strength_sources', 'mean_strength_sinks', 'viscosity']:
    print(f'{measure}:')
    for mode in modes:
        p = write_scientific(flux_props_stats_df[(flux_props_stats_df['measure'] == measure) & (flux_props_stats_df['mode'] == mode)]['p_bh'].values[0])
        print(f'{mode_dict[mode-2]}: $p_{{FDR}}=${p};')

# %%
for mode in modes:
    mean_area_sources_rest = flux_props_df[(flux_props_df['condition'] == 'Rest') & (flux_props_df['mode'] == mode)]['mean_area_sources'].values
    mean_area_sources_five = flux_props_df[(flux_props_df['condition'] == '5-MeO-DMT') & (flux_props_df['mode'] == mode)]['mean_area_sources'].values
    diff_area_sources = mean_area_sources_five - mean_area_sources_rest
    mean_strength_sources_rest = flux_props_df[(flux_props_df['condition'] == 'Rest') & (flux_props_df['mode'] == mode)]['mean_strength_sources'].values
    mean_strength_sources_five = flux_props_df[(flux_props_df['condition'] == '5-MeO-DMT') & (flux_props_df['mode'] == mode)]['mean_strength_sources'].values
    diff_strength_sources = mean_strength_sources_five - mean_strength_sources_rest
    mean_het_rest = order_df[(order_df['condition'] == 'Rest') & (order_df['mode'] == mode)]['H'].values
    mean_het_five = order_df[(order_df['condition'] == '5-MeO-DMT') & (order_df['mode'] == mode)]['H'].values
    diff_het = mean_het_five - mean_het_rest
    
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(2.4, 1.9))
    #ax.scatter(diff_area_sources, diff_het, alpha = .5, marker='2', color = 'black')
    sns.kdeplot(x=diff_area_sources, y=diff_het, ax=ax, cmap='viridis', alpha = 0.3, fill = True, levels=10, thresh=0.2)
    sns.regplot(x=diff_area_sources, y=diff_het, ax=ax, color='black', scatter=False, line_kws={"ls":"-.", "lw": 1})
    plt.xlabel(r'$\Delta$ Source Area ($cm^2$)')
    plt.ylabel(r'$\Delta H$')
    sns.despine()
    fact = 0.01
    ax.set_xlim([np.min(diff_area_sources) - fact * (np.max(diff_area_sources) - np.min(diff_area_sources)), np.max(diff_area_sources) + fact * (np.max(diff_area_sources) - np.min(diff_area_sources))])
    rho, p = stats.spearmanr(diff_area_sources, diff_het)
    if mode == 5:
        ax.text(0.05, 0.95, f'$rs=${np.round(rho, 2)}', ha='left', va='top', transform=ax.transAxes, fontsize=14)
    else:
        ax.text(0.95, 0.95, f'$rs=${np.round(rho, 2)}', ha='right', va='top', transform=ax.transAxes, fontsize=14)

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(2.4, 1.9))
    #ax.scatter(diff_area_sinks, diff_het, alpha = .5, marker='2', color = 'black')
    sns.kdeplot(x=diff_strength_sources, y=diff_het, ax=ax, cmap='viridis', alpha = 0.3, fill = True, levels=10, thresh=0.2)
    sns.regplot(x=diff_strength_sources, y=diff_het, ax=ax, color='black', scatter=False, line_kws={"ls":"-.", "lw": 1})
    plt.xlabel(r'$\Delta$ Source Strength ($s^{-1}$)')
    plt.ylabel(r'$\Delta H$')
    sns.despine()
    fact = 0.01
    ax.set_xlim([np.min(diff_strength_sources) - fact * (np.max(diff_strength_sources) - np.min(diff_strength_sources)), np.max(diff_strength_sources) + fact * (np.max(diff_strength_sources) - np.min(diff_strength_sources))])
    rho, p = stats.spearmanr(diff_strength_sources, diff_het)
    if mode == 5:
        ax.text(0.05, 0.95, f'$rs=${np.round(rho, 2)}', ha='left', va='top', transform=ax.transAxes, fontsize=14)
    else:
        ax.text(0.95, 0.95, f'$rs=${np.round(rho, 2)}', ha='right', va='top', transform=ax.transAxes, fontsize=14)

# %%


# %%
# get the alignment_df for each perp and condition and append them to a single df

for i, mode in enumerate(modes):
    for j, perp in enumerate(perps):
        for k, condition in enumerate(conditions):
            alignment_df = pd.read_csv(f'flow_fields/{perp}/{condition}/mode_{mode}/alignment_df.csv', index_col=0)
            if i == 0 and j == 0 and k == 0:
                align_df = alignment_df.copy()
            else:
                align_df = pd.concat([align_df, alignment_df], axis=0)
# save
align_df.to_csv('flow_fields/alignment_df.csv')

# do stats
alignment_stats_df = pd.DataFrame(columns=['mode', 'measure', 't', 'p', 'cil', 'ciu', 'd', 'bf', 'dof', 'power'])
for mode in modes:
    for measure in ['ge', 'n_coms', 'mod', 'negentropy']:
        rest = align_df[(align_df['condition'] == 'Rest') & (align_df['mode'] == mode)][measure].values
        five = align_df[(align_df['condition'] == '5-MeO-DMT') & (align_df['mode'] == mode)][measure].values
        t, p, cil, ciu, d, bf, dof, power = aug_t_test(five, rest)
        alignment_stats_df = alignment_stats_df.append({'mode': mode, 'measure': measure, 't': t, 'p': p, 'cil': cil, 'ciu': ciu, 'd': d, 'bf': bf, 'dof': dof, 'power': power}, ignore_index=True)
# correct p-values
alignment_stats_df['p_bh'] = multi.multipletests(alignment_stats_df['p'], method='fdr_bh')[1]

# save
alignment_stats_df.to_csv('flow_fields/alignment_stats_df.csv')

alignment_stats_df

# %%
# write stats for ge

for measure in ['ge', 'n_coms']:
    print(f'{measure}:')
    for mode in modes:
        p = write_scientific(alignment_stats_df[(alignment_stats_df['measure'] == measure) & (alignment_stats_df['mode'] == mode)]['p_bh'].values[0])
        print(f'{mode_dict[mode-2]}: $p_{{FDR}}=${p};')

# %%
for measure in ['ge', 'mod', 'negentropy']:
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    sns.violinplot(x='mode', y=measure, hue='condition', data=align_df, ax=ax,
                   split=True, linewidth=1.3, palette={"Rest": "green", "5-MeO-DMT": "purple"}, saturation=1, gap=0.1, alpha=1, cut=0, fill=False)
    ax.get_legend().remove()
    plt.xlabel('')
    ax.set_xticklabels([str(mode_dict[m-2]) for m in modes])
    if measure == 'ge':
        plt.ylabel('Temporal Global Efficiency')
    elif measure == 'n_coms':
        plt.ylabel('No. Temporal Communities')
        ax.set_yscale('log', base=10)
    elif measure == 'mod':
        plt.ylabel(r'Temporal Modularity ($Q$)')
    elif measure == 'negentropy':
        plt.ylabel('Degree Negentropy (bits)')
    # title

    for j, mode in enumerate(modes):
        max_val = align_df[align_df['mode'] == mode][measure].max()
        p= alignment_stats_df[(alignment_stats_df['mode'] == mode) & (alignment_stats_df['measure'] == measure)]['p_bh'].values[0]
        if p < 0.001:
            ax.text(j-0.12, max_val, '***', fontsize=20)
        elif p < 0.01:
            ax.text(j-0.09, max_val, '**', fontsize=20)
        elif p < 0.05:
            ax.text(j-0.06, max_val, '*', fontsize=20)

    sns.despine()

# %%
fig, ax = plt.subplots(1, 1, figsize=(4, 2))

# individual data points
for j, mode in enumerate(modes):
    rest = align_df[(align_df['condition'] == 'Rest') & (align_df['mode'] == mode)]['n_coms']
    five = align_df[(align_df['condition'] == '5-MeO-DMT') & (align_df['mode'] == mode)]['n_coms']
    for i in range(len(rest)):
        ax.plot(j-0.2, rest.values[i], 'o', color='green', alpha=0.2)
    for i in range(len(five)):
        ax.plot(j+0.2, five.values[i], 'o', color='purple', alpha=0.2)
# bar plot
sns.barplot(x='mode', y='n_coms', hue='condition', data=align_df, ax=ax, palette={"Rest": "green", "5-MeO-DMT": "purple"},
            alpha=.1, edgecolor='black', 
            capsize=0.1, errwidth=0.8, errcolor='black', linewidth=1.3,
            dodge=True)
# custom_lines = [plt.Line2D([0], [0], color='green', lw=1),
#                 plt.Line2D([0], [0], color='purple', lw=1)]
# ax.legend(custom_lines, ['Rest', '5-MeO'], loc='upper right', frameon=False)
ax.get_legend().remove()

plt.xlabel('')
ax.set_xticklabels([str(mode_dict[m-2]) for m in modes])
plt.ylabel('Temporal \n Communities')
ax.set_yscale('log', base=10)
sns.despine()

for j, mode in enumerate(modes):
    max_val = align_df[align_df['mode'] == mode]['n_coms'].max()
    p= alignment_stats_df[(alignment_stats_df['mode'] == mode) & (alignment_stats_df['measure'] == 'n_coms')]['p_bh'].values[0]
    if p < 0.001:
        ax.text(j-0.15, max_val, '***', fontsize=20)
    elif p < 0.01:
        ax.text(j-0.09, max_val, '**', fontsize=20)
    elif p < 0.05:
        ax.text(j-0.06, max_val, '*', fontsize=20)



# %%
# compute the mean degree and stats between rest and 5-MeO-DMT

degree_df = pd.DataFrame(columns=['perp', 'condition', 'mode', 'mean_degree'])
for mode in modes:
    for perp in perps:
        for condition in conditions:
            degree_hist = np.load(f'flow_fields/{perp}/{condition}/mode_{mode}/degree_hist.npy', allow_pickle=True)
            # find the mean degree
            degree = np.arange(len(degree_hist))
            mean_degree = np.sum(degree * degree_hist) / np.sum(degree_hist)
            degree_df = degree_df.append({'perp': perp, 'condition': condition, 'mode': mode, 'mean_degree': mean_degree}, ignore_index=True)
# save
degree_df.to_csv('flow_fields/degree_df.csv')

# do stats
degree_stats_df = pd.DataFrame(columns=['mode', 't', 'p', 'cil', 'ciu', 'd', 'bf', 'dof', 'power'])
for mode in modes:
    rest = degree_df[(degree_df['condition'] == 'Rest') & (degree_df['mode'] == mode)]['mean_degree'].values
    five = degree_df[(degree_df['condition'] == '5-MeO-DMT') & (degree_df['mode'] == mode)]['mean_degree'].values
    t, p, cil, ciu, d, bf, dof, power = aug_t_test(five, rest)
    degree_stats_df = degree_stats_df.append({'mode': mode, 't': t, 'p': p, 'cil': cil, 'ciu': ciu, 'd': d, 'bf': bf, 'dof': dof, 'power': power}, ignore_index=True)
# correct p-values
degree_stats_df['p_bh'] = multi.multipletests(degree_stats_df['p'], method='fdr_bh')[1]

# write the stats
for mode in modes:
    p = write_scientific(degree_stats_df[degree_stats_df['mode'] == mode]['p_bh'].values[0])
    print(f'{mode_dict[mode-2]}: $p_{{FDR}}=${p};')

# %%
# compute the mean and std degree hist for each perp

# load example
perp = perps[0]
condition = conditions[0]
mode = modes[0]
degree_hist = np.load(f'flow_fields/{perp}/{condition}/mode_{mode}/degree_hist.npy', allow_pickle=True)
n_bins = len(degree_hist)

for mode in modes:
    rest_deg_hist = np.zeros((n_bins, len(perps)))
    five_deg_hist = np.zeros((n_bins, len(perps)))
    for i, perp in enumerate(perps):
        for j, condition in enumerate(conditions):
            degree_hist = np.load(f'flow_fields/{perp}/{condition}/mode_{mode}/degree_hist.npy', allow_pickle=True)
            if condition == 'Rest':
                rest_deg_hist[:, i] = degree_hist
            else:
                five_deg_hist[:, i] = degree_hist
    # get the mean and se
    rest_mean = np.mean(rest_deg_hist, axis=1)
    rest_se = np.std(rest_deg_hist, axis=1) / np.sqrt(len(perps))
    five_mean = np.mean(five_deg_hist, axis=1)
    five_se = np.std(five_deg_hist, axis=1) / np.sqrt(len(perps))

    # print max degree
    # i.e max non-zero degree
    rest_max_deg = np.max(np.nonzero(rest_mean))
    print(f'{mode_dict[mode-2]}: Max degree: {rest_max_deg};')
    five_max_deg = np.max(np.nonzero(five_mean))
    print(f'{mode_dict[mode-2]}: Max degree: {five_max_deg};')

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 2))
    ax.plot(np.arange(n_bins), rest_mean, color='green', linestyle='-', label='Rest')
    #ax.fill_between(np.arange(n_bins), rest_mean - rest_se, rest_mean + rest_se, color='green', alpha=0.1)
    ax.plot(np.arange(n_bins), five_mean, color='purple', linestyle='-', label='5-MeO-DMT')
    #ax.fill_between(np.arange(n_bins), five_mean - five_se, five_mean + five_se, color='purple', alpha=0.1)
    plt.xlabel('Degree')
    plt.ylabel('Freq')
    if mode == 3 or mode == 4:
        set_xlim = 550
        plt.xlabel('')
        if mode == 3:
            plt.xticks([])
            # add legend with no border
            ax.legend(loc='lower right', frameon=False)
    elif mode == 5:
        set_xlim = 5500
    ax.set_xlim(0, set_xlim)
    sns.despine()
    ax.set_yscale('log')
    # add text in the top right to say the mode
    ax.text(0.8, 0.95, f'{mode_dict[mode-2]}', transform=ax.transAxes, fontsize=14, verticalalignment='top')


# %%
sing_dyn_df

# %%
sing_dyn_df = pd.DataFrame(columns=['subject', 'condition', 'mode', 'wave_type', 'number', 'mean_duration', 'std_duration', 'sum_duration'])

for perp in perps:
    for condition in conditions:
        for mode in modes:
            wprops_df = pd.read_csv(f'flow_fields/{perp}/{condition}/mode_{mode}/dynamics/wprops_df.csv', index_col=0)
            wprops_df['subject'] = perp; wprops_df['condition'] = condition; wprops_df['mode'] = mode
            wprops_df = wprops_df[['subject', 'condition', 'mode', 'wave_type', 'number', 'mean_duration', 'std_duration', 'sum_duration']]
            sing_dyn_df = pd.concat([sing_dyn_df, wprops_df])
sing_dyn_df = sing_dyn_df.reset_index(drop=True)

sing_dyn_df = sing_dyn_df[sing_dyn_df['wave_type'] != 'Unknown']
sing_dyn_df['wave_type'] = sing_dyn_df['wave_type'].replace(wave_dict.values(), wave_dict.keys())
sing_dyn_df['wave_type'] = sing_dyn_df['wave_type'].replace(abbrev_dict)

sing_dyn_df.to_csv('flow_fields/sing_dyn_df.csv', index=False)

sing_dyn_stats_df = pd.DataFrame(columns=['mode', 'wave_type', 'measure',
                                          't', 'p', 'cil', 'ciu', 'd', 'bf', 'dof', 'power'])

measures = ['mean_duration', 'sum_duration', 'number']

for mode in modes:
    for measure in measures:
        for wave_type in sing_dyn_df['wave_type'].unique():
            rest = sing_dyn_df[(sing_dyn_df['condition'] == 'Rest') & (sing_dyn_df['mode'] == mode) & (sing_dyn_df['wave_type'] == wave_type)][measure].values
            five = sing_dyn_df[(sing_dyn_df['condition'] == '5-MeO-DMT') & (sing_dyn_df['mode'] == mode) & (sing_dyn_df['wave_type'] == wave_type)][measure].values
            rest = rest.astype(float)
            five = five.astype(float)
            t, p, cil, ciu, d, bf, dof, power = aug_t_test(five, rest)
            sing_dyn_stats_df = sing_dyn_stats_df.append({'mode': mode, 'wave_type': wave_type, 'measure': measure,
                                                          't': t, 'p': p, 'cil': cil, 'ciu': ciu, 'd': d, 'bf': bf, 'dof': dof, 'power': power}, ignore_index=True)
# correct p-values
sing_dyn_stats_df['p_bh'] = multi.multipletests(sing_dyn_stats_df['p'], method='fdr_bh')[1]

# save
sing_dyn_stats_df.to_csv('flow_fields/sing_dyn_stats_df.csv')

sing_dyn_stats_df

# %%
# for 'mean_duration# print the max t-value and associated statistics for each mode

for mode in modes:
    max_T = 0
    for wave_type in sing_dyn_df['wave_type'].unique():
        t = sing_dyn_stats_df[(sing_dyn_stats_df['mode'] == mode) & (sing_dyn_stats_df['wave_type'] == wave_type) & (sing_dyn_stats_df['measure'] == 'mean_duration')]['t'].values[0]
        if abs(t) > abs(max_T):
            max_T = t
            max_wave_type = wave_type
    T = write_scientific(max_T)
    p = write_scientific(sing_dyn_stats_df[(sing_dyn_stats_df['mode'] == mode) & (sing_dyn_stats_df['wave_type'] == max_wave_type) & (sing_dyn_stats_df['measure'] == 'mean_duration')]['p_bh'].values[0])
    bf = write_scientific(sing_dyn_stats_df[(sing_dyn_stats_df['mode'] == mode) & (sing_dyn_stats_df['wave_type'] == max_wave_type) & (sing_dyn_stats_df['measure'] == 'mean_duration')]['bf'].values[0])
    print(f'{mode_dict[mode-2]} max: {max_wave_type}: $T=${T}, $p_{{FDR}}=${p}, $BF=${bf};')

# %%
for mez, measure in enumerate(measures):
    # make 3 subplots (one for each mode) that has the wave_type on the x-axis and the measure on the y-axis
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.5))
    for i, mode in enumerate(modes):
        if mez == 0:
            width = 1
        else:
            width = 0.8
        sns.violinplot(x='wave_type', y=measure, hue='condition', data=sing_dyn_df[(sing_dyn_df['mode'] == mode)], ax=ax[i], width=width,
                        split=True, linewidth=1.3, palette={"Rest": "green", "5-MeO-DMT": "purple"}, saturation=1, gap=0.1, alpha=1, cut=0, fill=False)
        ax[i].get_legend().remove()
        ax[i].set_xlabel('')
        ax[i].set_title(f'{mode_dict[mode-2]}', pad=10, fontsize=14)
        ax[i].set_xticklabels(labels=ax[i].get_xticklabels(),
            rotation=45, ha='right')
        x_order = [item.get_text() for item in ax[0].get_xticklabels()]
        for j, wave_type in enumerate(x_order):
            max_val = sing_dyn_df[(sing_dyn_df['mode'] == mode) & (sing_dyn_df['wave_type'] == wave_type)][measure].max()
            p = sing_dyn_stats_df[(sing_dyn_stats_df['mode'] == mode) & (sing_dyn_stats_df['wave_type'] == wave_type) & (sing_dyn_stats_df['measure'] == measure)]['p_bh'].values[0]
            if p < 0.001:
                ax[i].text(j-0.18, max_val, '***', fontsize=20)
            elif p < 0.01:
                ax[i].text(j-0.15, max_val, '**', fontsize=20)
            elif p < 0.05:
                ax[i].text(j-0.12, max_val, '*', fontsize=20)

        if measure == 'number':
            ax[i].set_ylabel('No. Patterns')
        elif measure == 'mean_duration':
            ax[i].set_ylabel('Pattern Endurance (ms)')
        elif measure == 'sum_duration':
            ax[i].set_ylabel('Total Duration (ms)')
        sns.despine()
    plt.tight_layout()

# %%



