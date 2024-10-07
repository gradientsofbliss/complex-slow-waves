"""
29.06.2024

Tools for complex wave pattern analysis in M/EEG.

@authors: George Blackburne, Marco Fabus

"""

import numpy as np
import pandas as pd

def emd_imfs(timeseries, sfreq, max_imfs=6,
             save=False, **kwargs):
    '''
    Get intrinsic mode functions and their instantaneous properties
    via iterative mask sift empirical mode decomposition (EMD).

    https://doi.org/10.1152/jn.00315.2021

    Parameters
    ----------
    timeseries : 1D array
        The timeseries to compute the alpha power of
    sfreq : int
        The sampling frequency in Hz
    max_imfs : int
        The maximum number of imfs to compute
    save : bool
        Whether to save the imfs and instantaneous properties

    Returns
    -------
    imfs : 2D array
        The intrinsic mode functions
    IP : 2D array
        The instantaneous phase
    IF : 2D array
        The instantaneous frequency
    IA : 2D array
        The instantaneous amplitude
        
    '''

    import emd
    import os

    imfs = emd.sift.iterated_mask_sift(timeseries, max_imfs=max_imfs)
    
    # # chuck faulty drift imfs
    # if np.isnan(imfs[0]).any():
    #     bad_index = np.where(np.isnan(imfs[0]))[0]
    #     imfs = imfs[:,:bad_index[0]]

    IP, IF, IA = emd.spectra.frequency_transform(imfs, sfreq, 'nht')

    if save:
        if not os.path.exists(kwargs['path']):
            os.makedirs(kwargs['path'])
        np.save(kwargs['path']+'imfs.npy', imfs)
        np.save(kwargs['path']+'IP.npy', IP)
        np.save(kwargs['path']+'IF.npy', IF)
        np.save(kwargs['path']+'IA.npy', IA)

    return imfs, IP, IF, IA

def emd_spectra(IF, IA, 
                low_freq=0.5, high_freq=50, res=297,
                form='linear'):
    '''
    Hilbert-Huang Transform of EMD power spectra

    https://doi.org/10.1098/rspa.1998.0193

    Parameters
    ----------
    IF : array
        The instantaneous frequency
    IA : array
        The instantaneous amplitude
    low_freq : float
        The low frequency bound
    high_freq : float
        The high frequency bound
    res : int
        The resolution of the spectra
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

    freq_edges, freq_centres = emd.spectra.define_hist_bins(low_freq, high_freq, res, 'linear')
    f, hht = emd.spectra.hilberthuang(IF, IA, freq_edges, scaling='density')

    if form == 'linear':
        psd = hht
    if form == 'log':
        psd = np.log10(hht)
    elif form == 'dB':
        psd = 10 * np.log10(hht)

    return psd, freq_centres

def smooth_spatial(IP, chs, sfreq, ch_pos, input='IP',
                         res=32, kernel_size=1, out_wrapped=True, 
                         smooth_wrapped=False, nas_in=0.35, raw=None):
    """

    Pushes phase or amplitude onto grid and smooths spatially across scalp for a single mode,
    with radial basis function interpolation and a multidimensional gaussian kernel.

    Parameters
    ----------
    IP : np.array
        The instantaneous phase or amplitude
        for a single mode, all channels stacked.
    chs : list
        The channel names.
    sfreq : int
        The sample rate.
    ch_pos : np.array
        The channel positions.
    input : str, optional
        The input mode. The default is 'IP'.
    res : int, optional
        The resolution. The default is len(chs).
    kernel_size : int, optional
        Width of gaussian kernel.
    out_wrapped : bool, optional
        If the output should be wrapped. The default is True.
    smooth_wrapped : bool, optional
        If the input should be smoothed. The default is False.
    nas_in : float, optional
        The nasion. The default is 0.35.
    raw : mne.io.Raw, optional
        The raw data. The default is None.

    Returns
    -------
    out: np.array
        The smoothed phase or amplitude.
    X: np.array
        The x grid.
    Y: np.array
        The y grid.

    """
    import mne
    import emd
    from scipy import ndimage
    from scipy.interpolate import Rbf

    N_ch = len(chs)

    if res is None:
        res = N_ch

    N_s = IP.shape[1]
    IP_all = IP

    if smooth_wrapped:
        IP_all = IP_all % (2 * np.pi)

    IP_all_smooth = np.zeros((N_s, res, res))*np.nan
    d0 = 0.8*nas_in # scalp diameter
    xgrid = np.mgrid[-d0/2:d0/2:res*1j, -d0/2:d0/2:res*1j] # grid for interpolation
    Y, X = xgrid
    circle = (X**2 + Y**2) < (d0/2)**2 # scalp circle
    xflat = xgrid.reshape(2, -1).T # flattened grid

    for i in range(N_s):

        # Smooth with radial basis function with gaussian kernel
        rbf = Rbf(ch_pos[:, 0], ch_pos[:, 1], IP_all[:, i])
        IPflat = rbf(xflat[:, 0], xflat[:, 1])
        IPgrid = IPflat.reshape(res, res)
        IPgrid[~circle] = np.nan
        Z = IPgrid.copy()
        # pos = ch_pos
        # outlines = None

        if kernel_size == 0:
            Z[np.isnan(Z)] = 0
            Z_smooth = Z
        else:
            V = Z.copy()
            V[np.isnan(Z)]=0
            VV = ndimage.gaussian_filter(V, kernel_size, mode='nearest')

            W = 0*Z.copy()+1
            W[np.isnan(Z)] = 0
            WW = ndimage.gaussian_filter(W, kernel_size, mode='nearest')

            Z_smooth = VV/WW

        IP_all_smooth[i, :, :] = Z_smooth

    if out_wrapped:
        out = (IP_all_smooth) % (2 * np.pi)
    else:
        out = IP_all_smooth

    # ensure that pixels outside the scalp are nans for all samples
    out[:, ~circle] = np.nan

    return out, X, Y

def kuramoto_order_parameter(IP):
    """
    Calculates a Kuramoto (adjacent) parameter for signal.

    $R(t)=\frac{1}{N}\left|\sum_{j=1}^N e^{i \theta_j}\right|$

    Parameters
    ----------
    IP : N_s x res x res ndarray.

    Returns
    -------
    R : N_s ndarray.
        The Kuramoto order parameter for each sample.

    """
    
    if len(IP.shape) == 3:
        resX = IP.shape[-1]
        resY = IP.shape[-2]
        N_s = IP.shape[0]

    if len(IP.shape) == 2:
        resX = IP.shape[0]
        resY = IP.shape[1]
        N_s = 1

    if len(IP.shape) == 1:
        resX = IP.shape[0]
        resY = 1
        N_s = 1
        
    S = resX * resY
    R = np.zeros(N_s)
    IP_flat = np.reshape(IP, (N_s, resX*resY))

    IP_flat = IP_flat[:, ~np.isnan(IP_flat).any(axis=0)] # get rid of elements outside the scalp
    
    R = 1 / S * np.abs(np.sum(np.exp(1j * IP_flat), axis=1))
        
    return R

def global_coherence(R):
    return np.mean(R)
def global_metastability(R):
    return np.var(R)

def local_pixels(nas_in, res, kernel=6):
    d0 = 0.8*nas_in # scalp diameter
    xgrid = np.mgrid[-d0/2:d0/2:res*1j, -d0/2:d0/2:res*1j] # grid for interpolation
    Y, X = xgrid
    circle = (X**2 + Y**2) < (d0/2)**2 # scalp circle
    in_pix = np.argwhere(circle)
    pix_coords = []
    for x,y in in_pix:
        # if the kernel width ends up outside the scalp then just take the area that is inside the scalp
        area_pix = circle[x-(kernel if x-kernel >= 0 else x):
                          x+(kernel if x+kernel < res else np.diff([x, res])[0]),
                          y-(kernel if y-kernel >= 0 else y):
                          y+(kernel if y+kernel < res else np.diff([y, res])[0])]
        # if more than 50% of the area is in the circle then append the coords
        if np.sum(area_pix) / (2*kernel+1)**2 > 0.5:
            pix_coords.append((x, y)) 
    return pix_coords

def local_kuramoto(IP, kernel=4, inner_grid_coords=None):
    """
    Computes spatially local Kuramoto properties.

    Parameters
    ----------
    IP : N_s x res x res ndarray.
    kernel : int, optional
        How many pixels to include in the kernel. The default is 6.

    Returns
    -------
    R_l_avg : N_s ndarray.
        The local Kuramoto order parameter for each sample.

    """

    if inner_grid_coords is None:
        inner_grid_coords = local_pixels(0.35, IP.shape[-1], kernel=kernel)

    N_s = IP.shape[0] if len(IP.shape) == 3 else 1

    R_l = np.full((N_s, len(inner_grid_coords)), np.nan)

    for i in range(N_s):

        for j, (x, y) in enumerate(inner_grid_coords):
            # take the pixels in the kernel width around the current pixel
            # if the kernel width ends up outside the scalp then just take the area that is inside the scalp
            IP_area = IP[i, x-(kernel if x-kernel >= 0 else x):x+(kernel if x+kernel < IP.shape[-1] else np.diff([x, IP.shape[-1]])[0]),
                            y-(kernel if y-kernel >= 0 else y):y+(kernel if y+kernel < IP.shape[-2] else np.diff([y, IP.shape[-2]])[0])]

            R = kuramoto_order_parameter(IP_area)

            R_l[i, j] = np.mean(R)

    local_cohs= np.mean(R_l, axis=0)
    local_mets= np.var(R_l, axis=0)

    local_coh = np.mean(local_cohs)
    local_met = np.mean(local_mets)

    topvar_coh = np.var(local_cohs)
    topvar_met = np.var(local_mets)

    return local_coh, local_met, topvar_coh, topvar_met

def get_velocity_field(IP, sfreq, nas_in=0.35):
    """
    Compute the velocity field from the instantaneous phase (or amplitude)

    Parameters
    ----------
    IP : np.array
        The instantaneous phase.
    sfreq : int
        The sample rate.
    nas_in : float, optional
        The nasion. The default is 0.35.

    Returns
    -------
    u : np.array
        The x velocity.
    v : np.array
        The y velocity.
    speed : np.array
        The speed.

    """
    
    res = IP.shape[-1] 
    grad = np.gradient(IP, 0.8*nas_in/res, axis=[1, 2]) # spatial derivative
    dphi_dt = np.abs(np.gradient(IP, 1/sfreq, edge_order=1, axis=[0])) # absolute temporal derivative
    gX = grad[1]; gY = grad[0] # x and y components of spatial derivative
    u = -1 * dphi_dt * gX / (gX**2 + gY**2) # x velocity
    v = -1 * dphi_dt * gY / (gX**2 + gY**2) # y velocity
    speed = np.sqrt(u**2 + v**2)
    
    return u, v, speed

def avg_norm_vel(u, v, flatten=True,
                 smooth=False, smooth_window=10): 
    '''
    Compute the average velocity across the scalp 
    (i.e the directionality or collective motion of the field)

    Eq 7 from https://doi.org/10.1038/s41467-019-12918-8
    $\Phi=\frac{\left\|\sum_{\mathrm{i}=1}^N \overrightarrow{\mathbf{v}}_{\mathrm{i}}\right\|}{\sum_{i=1}^N\left\|\overrightarrow{\mathbf{v}_{\mathrm{i}}}\right\|}$

    Parameters
    ----------
    U : np.array
        The x velocity.
    V : np.array
        The y velocity.
    flatten : bool, optional
        Whether to flatten the data. The default is True.
    smooth : bool, optional
        Whether to smooth the data. The default is False.
    smooth_window : int, optional
        The window for smoothing. The default is 10.

    Returns
    -------
    phi : np.array
        The average velocity
        1D array of length n_timepoints 

    '''

    from scipy.signal import savgol_filter

    if flatten:
        u_flat = u.reshape(u.shape[0], -1)
        v_flat = v.reshape(v.shape[0], -1)
        u_flat = u_flat[:, ~np.isnan(u_flat).any(axis=0)]
        v_flat = v_flat[:, ~np.isnan(v_flat).any(axis=0)]
    else:
        u_flat = u
        v_flat = v
        
    su = np.nansum(u_flat, axis=1) # sum of x velocities
    sv = np.nansum(v_flat, axis=1) # sum of y velocities
    num = np.sqrt(su**2 + sv**2) # magnitude of the sum of velocities
    abs_w = np.sqrt(u_flat**2 + v_flat**2) # magnitudes of velocities (i.e speed)
    den = np.nansum(abs_w, axis=1) # sum of magnitudes of velocities
    
    phi = num / den 
    
    if smooth:
        phi_smooth = savgol_filter(phi, smooth_window, 3)
        return phi, phi_smooth
    else:
        return phi
    
def field_hetero(speed, flatten=True):
    """
    Calculate velocity field heterogeneity
    i.e the standard deviation of the velocity across all pixels at each timepoint

    Eq 8 from https://doi.org/10.1038/s41467-019-12918-8
    $H=\frac{1}{T} \sum_{t=1}^{t=T} \frac{1}{N_t} \sum_{\mathrm{i}=1}^{\mathrm{i}=N_t}\left[\nu_{\mathrm{i}}(t)-\mu(t)\right]^2$

    """

    if flatten:
        speed_flat = speed.reshape(speed.shape[0], -1)
        speed_flat = speed_flat[:, ~np.isnan(speed_flat).any(axis=0)]

    H = np.std(speed_flat, axis=1)    

    return H


def velocity_alignment(U, V, flatten=True, option='corr'):
    '''
    Compute recurrent velocity field patterns

    https://doi.org/10.1038/nn.3499 
    https://doi.org/10.1038/s41467-019-08999-0

    Parameters
    ----------
    U : np.array
        The x velocity.
    V : np.array
        The y velocity.
    flatten : bool, optional
        Whether to flatten the data. The default is True.
    option : str, optional
        The option for alignment. The default is 'dot'.

    Returns
    -------
    dot_mat : np.array
        The average velocity alignment.

    '''

    if flatten:
        U = U.reshape(U.shape[0], -1)
        V = V.reshape(V.shape[0], -1)
        U = U[:, ~np.isnan(U).any(axis=0)]
        V = V[:, ~np.isnan(V).any(axis=0)]

    if option == 'dot':
        dot_mat = np.dot(U, V.T).astype(np.float32) / U.shape[1]
    elif option == 'corr':
        dot_mat = np.corrcoef(np.concatenate((U, V), axis=1)).astype(np.float32)
        dot_mat = -np.log(1-(dot_mat**2))/2

    np.fill_diagonal(dot_mat, 0)
    
    return dot_mat

def shuffle_velocity_align(U, V, flatten=True, option='dot', nperm=1000):
    '''
    Null velocity field alignment

    Parameters
    ----------
    U : np.array
        The x velocity.
    V : np.array
        The y velocity.
    flatten : bool, optional
        Whether to flatten the data. The default is True.
    option : str, optional
        The option for alignment. The default is 'dot'.
    nperm : int, optional
        The number of permutations. The default is 1000.
    
    Returns
    -------
    max_perm : np.array
        The maximum value of the null distribution

    '''
    
    if flatten:
        U = U.reshape(U.shape[0], -1)
        V = V.reshape(V.shape[0], -1)
        U = U[:, ~np.isnan(U).any(axis=0)]
        V = V[:, ~np.isnan(V).any(axis=0)]

    maxes = []

    for perm in range(nperm):
        u_perm = np.apply_along_axis(np.random.permutation, 1, U)
        v_perm = np.apply_along_axis(np.random.permutation, 1, V)
        if option == 'dot':
            dot_mat = np.dot(u_perm, v_perm.T).astype(np.float32) / U.shape[1]
        elif option == 'corr':
            dot_mat = np.corrcoef(np.concatenate((u_perm, v_perm), axis=1)).astype(np.float32)
            dot_mat = -np.log(1-(dot_mat**2))/2
        
        np.fill_diagonal(dot_mat, 0)
        maxes.append(np.percentile(dot_mat, 99))
    
    max_perm = np.max(maxes)

    return dot_mat, max_perm

def thresh_matrix(dot_mat, max_perm):
    '''
    Threshold the velocity alignment matrix

    Parameters
    ----------
    dot_mat : np.array
        The velocity alignment.
    max_perm : np.array
        The permutation 99th percentile.

    Returns
    -------
    sig_mat : np.array
        The thresholded velocity alignment.

    '''
    sig_al = np.abs(dot_mat) > np.abs(max_perm)
    sig_mat = dot_mat.copy()
    sig_mat[~sig_al] = 0
    sig_mat[sig_al] = 1
    np.fill_diagonal(sig_mat, 1)
    sig_mat = sig_mat.astype(int)

    return sig_mat

def matrix_to_graph(matrix, threshold=0):
    '''
    Convert a matrix into a graph.

    Parameters
    ----------
    matrix : np.array
        The matrix.
    threshold : float, optional
        The threshold. The default is unthresholded.

    Returns
    -------
    G : nx.Graph
        The graph.

    '''
    
    import networkx as nx
    
    G = nx.Graph()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.abs(matrix[i, j]) > threshold:
                G.add_edge(i, j, weight=matrix[i, j])
    G.remove_edges_from(nx.selfloop_edges(G))

    return G

def degree_dist(G):
    import networkx as nx
    degree_sequence = [d for n, d in G.degree()]
    degree_hist = np.histogram(degree_sequence, bins=range(len(degree_sequence)+1))[0]
    return degree_hist
def dist_negentropy(degree_hist):
    import scipy.stats as stats
    max_ent = np.log2(len(degree_hist))
    ent = stats.entropy(degree_hist, base=2)
    return max_ent - ent
def global_efficiency(G):
    import networkx as nx
    ge = nx.global_efficiency(G)
    return ge
def community_detection(G):
    from networkx.algorithms import community
    comms = community.greedy_modularity_communities(G)
    return comms
def modularity(G, comms):
    import networkx as nx
    mod = nx.community.modularity(G, comms)
    return mod

def find_singularities_divergence(u, v, mask=None, nas_in=0.35, upsampling=4,
                                  extent=[-1, 1], nlevels=10, robust_levels=False,
                                  robust_th=.01, upsample_vel=False):
    """
    Find singularities in flow as extreme points of divergence.

    Parameters
    ----------
    u : np.array
        The x velocity.
    v : np.array
        The y velocity.
    mask : np.array, optional
        The mask. The default is None.
    nas_in : float, optional
        The nasion. The default is 0.35.
    upsampling : int, optional
        The upsampling factor. The default is 4.
    extent : list, optional
        The extent. The default is [-1, 1].
    nlevels : int, optional
        The number of levels. The default is 10.
    robust_levels : bool, optional
        If the levels should be robust. The default is False.
    robust_th : float, optional
        The threshold for robust levels. The default is .01.
    upsample_vel : bool, optional
        If the velocity should be upsampled. The default is False.

    Returns
    -------
    out : list
        The singularities.

    """

    import cv2 as cv
    from skimage import transform, measure
    import itertools

    res = u.shape[-1]
    
    if upsample_vel:
        res *= upsampling
        
    N_s = u.shape[0]
    
    out = []

    speed = np.sqrt(u**2 + v**2)
    
    for s in range(N_s):
        
        if upsample_vel:
            # Alternative: upsample velocity field, not divergence
            uu = transform.pyramid_expand(u[s, :, :], upscale=upsampling)
            vv = transform.pyramid_expand(v[s, :, :], upscale=upsampling)
        else:
            uu = u[s, :, :]
            vv = v[s, :, :]
        
        # Find partial derivatives of each velocity component
        gradU = np.gradient(uu, 0.8*nas_in/res, edge_order=1, axis=[0, 1])
        gradV = np.gradient(vv, 0.8*nas_in/res, edge_order=1, axis=[0, 1])
        gUx = gradU[1] # i.e how quickly is the x velocity changing in the x direction
        gVy = gradV[0] # i.e how quickly is the y velocity changing in the y direction
        
        # Compute divergence, mask outside scalp
        divV = gUx + gVy
        divVnan = divV.copy(); divVnan[~mask] = np.nan
        
        # Upsample for singularity detection
        im = transform.pyramid_expand(divV, upscale=upsampling)
        res_us = res * upsampling
        x = np.linspace(extent[0], extent[1], res_us)
        y = np.linspace(extent[0], extent[1], res_us)
        Xus, Yus = np.meshgrid(x, y)
        Rus = np.sqrt(Xus**2 + Yus**2)
        mask_us = Rus < 1
                
        # Define contour levels
        if robust_levels:
            lev_min = np.nanpercentile(divVnan, robust_th*100)
            lev_max = np.nanpercentile(divVnan, 100-robust_th*100)
            levels = np.linspace(lev_min, lev_max, nlevels)
        else:
            levels = np.linspace(np.nanmin(divVnan), np.nanmax(divVnan), nlevels)
            
        # Find closed curves at nlevels
        cs = []
        for l in levels:
            contours = measure.find_contours(im, mask=mask_us, level=l)
            c = [x for x in contours if np.all(x[0]==x[-1])]
            cs.append(c)
        
        # Find only closed curves embedded within closed curve of level above / below
        nested_cs = []
        nested_cs_l = [] # Divergence levels
        for l in range(nlevels): 
            conts = cs[l]
            N_c = list(range(len(conts)))
            
            if len(N_c) == 0: continue
            
            if levels[l] < 0: # Sinks, closed curve level above
                N_cp1 = list(range(len(cs[l+1])))
                conts_above = cs[l+1]
                for i, j in list(itertools.product(N_c, N_cp1)):
                    val = np.all(measure.points_in_poly(conts[i], conts_above[j]))
                    if val:
                        nested_cs.append(conts[i])
                        nested_cs_l.append(levels[l])
            
            if levels[l] > 0: # Sources, closed curve level below
                N_cm1 = list(range(len(cs[l-1])))
                conts_below = cs[l-1]
                for i, j in list(itertools.product(N_c, N_cm1)):
                    val = np.all(measure.points_in_poly(conts[i], conts_below[j]))
                    if val:
                        nested_cs.append(conts[i])
                        nested_cs_l.append(levels[l])
            
        # Remove if any curve still inside another
        rg1 = list(range(len(nested_cs)))
        rg2 = list(range(len(nested_cs) - 1))
        keep = np.ones(len(nested_cs))
        for i, j in list(itertools.product(rg1, rg2)):
            if i != j:
                val = np.all(measure.points_in_poly(nested_cs[i], nested_cs[j]))
                if val:
                    keep[j] = 0
                    continue
        keep = list(keep)
        nested_cs_uniq = [val for i, val in enumerate(nested_cs) if keep[i]==1]
        nested_cs_l_uniq = [val for i, val in enumerate(nested_cs_l) if keep[i]==1]
                
        # Extract source / sink info: transformed coordinates, centroid, area, strength
        sings = np.zeros((len(nested_cs_uniq), 5), dtype=object)
        for ix, contour in enumerate(nested_cs_uniq):
            # Transform to original image
            tr_c = (2*contour - (res_us-1)) / (res_us-1)
            centroid = np.mean(tr_c, axis=0)
            strength = nested_cs_l_uniq[ix]
            sing_type = int(strength > 0) # 0 = sink, 1 = source
            
            c = np.expand_dims(tr_c.astype(np.float32), 1)
            c = cv.UMat(c)
            area = cv.contourArea(c)
            
            props = [centroid, tr_c, strength, area, sing_type]
            sings[ix, :] = props
            
        out.append(sings)
    
    return out

def sings_to_df(sings, idx, times, speeds, extent=[-1, 1], res=32):
    """
    Convert the singularities to a dataframe.
    """

    rows_list = []
    for i, s in enumerate(sings):
        N_sing = s.shape[0]
        ix = idx[i]
        t = times[ix]
        speed = speeds[i]
        for j in range(N_sing):
            x = s[j][0][1]
            y = s[j][0][0]
            strength = s[j][2]
            area = s[j][3]
            typ = s[j][4]
            flux = np.sqrt(x**2 + y**2)
            dict1 = {'time':t, 'avg_speed':speed,
                     'IA_frame':ix, 'x':x, 'y':y, 
                     'strength':strength, 'area':area, 
                     'type':typ}
            rows_list.append(dict1)
    
    df = pd.DataFrame(rows_list) 

    df.loc[df['type']==0, 'strength'] = df.loc[df['type']==0, 'strength'] * (-1) # make strength *-1 for sinks
    df['area'] = df['area'] * 196 # convert area to cm^2, assuming r=14cm
    df['flux'] = df['area'] * df['strength']

    # convert x and y to back to the original grid space
    ext_lin = np.linspace(extent[0], extent[1], res)
    df['x_grid'] = df['x']
    df['y_grid'] = df['y']
    for i in range(len(df)):
        df.loc[i, 'x_grid'] = int(np.abs(ext_lin - df.loc[i, 'x']).argmin())
        df.loc[i, 'y_grid'] = int(np.abs(ext_lin - df.loc[i, 'y']).argmin())

    return df

def flux_props(singularities_df):
    '''
    Compute flux properties

    Parameters
    ----------
    singularities_df: DataFrame
    
    Returns
    -------
    n_sources: int
    n_sinks: int
    area_sources: float
    area_sinks: float
    strength_sources: float
    strength_sinks: float
    viscosity: float
    ss_a_asym: float
        asymmetry in source-sink area
        -Positive values indicate source dominance, 
        -Negative values indicate sink dominance
    ss_f_asym: float
        asymmetry in flux of sources and sinks
        -Positive values indicate source dominance, 
        -Negative values indicate sink dominance
    
    '''
    
    singularities_df = singularities_df.replace([np.inf, -np.inf], np.nan).dropna()

    n_sources = singularities_df[singularities_df['type']==1].shape[0]
    n_sinks = singularities_df[singularities_df['type']==0].shape[0]

    area_sources = singularities_df[singularities_df['type']==1]['area'].mean()
    area_sinks = singularities_df[singularities_df['type']==0]['area'].mean()
    ss_a_asym = (area_sources - area_sinks) / (area_sources + area_sinks)

    strength_sources = singularities_df[singularities_df['type']==1]['strength'].mean()
    strength_sinks = singularities_df[singularities_df['type']==0]['strength'].mean()

    flux_sources = singularities_df[singularities_df['type']==1]['flux'].mean()
    flux_sinks = singularities_df[singularities_df['type']==0]['flux'].mean()
    ss_f_asym = (flux_sources - flux_sinks) / (flux_sources + flux_sinks)

    viscosity = singularities_df['flux'].mean()
    
    return n_sources, n_sinks, area_sources, area_sinks, strength_sources, strength_sinks, viscosity, ss_a_asym, ss_f_asym

def jacobian(u, v, nas_in=0.35):
    res = u.shape[-1]
    Du = np.gradient(u, 0.8*nas_in/res, edge_order=1, axis=[1, 2])
    Dv = np.gradient(v, 0.8*nas_in/res, edge_order=1, axis=[1, 2])
    du_dx = Du[1]; du_dy = Du[0]; dv_dx = Dv[1]; dv_dy = Dv[0]
    det = du_dx*dv_dy - dv_dx*du_dy
    trace = du_dx + dv_dy
    return det, trace

wave_dict = {0: 'Unknown',
             1: 'Stable Node',
             2: 'Stable Focus',
             3: 'Unstable Node',
             4: 'Unstable Focus',
             5: 'Saddle'
             }

wave_dict_ext = wave_dict.copy()
wave_dict_ext[6] = 'Standing'; wave_dict_ext[7] = 'Plane'

def singularity_taxonomy(u, v, singularities_df):
    """
    Taxonomise the waves based on the Jacobian

    https://doi.org/10.1371/journal.pcbi.1006643

    Parameters
    ----------
    u : np.array
        The x velocity.
    v : np.array
        The y velocity.
    singularities_df : pd.DataFrame
        The singularities dataframe.

    Returns
    -------
    singularities_df : pd.DataFrame
        The singularities dataframe.

    """

    singularities_df['wave_type'] = 0

    res = u.shape[-1]

    det, trace = jacobian(u, v)

    for i in range(len(singularities_df)):
        
        frame = singularities_df.loc[i, 'IA_frame']
        phi = singularities_df.loc[i, 'avg_speed']
        x_ = singularities_df.loc[i, 'x_grid']
        y_ = singularities_df.loc[i, 'y_grid']

        nei_co = [(x_-1, y_), (x_+1, y_), (x_, y_-1), (x_, y_+1)] # neighbours
        nei_co  = [(int(x), int(y)) for x, y in nei_co]

        det_ = np.nanmean([det[frame][x, y] for x, y in nei_co])
        trace_ = np.nanmean([trace[frame][x, y] for x, y in nei_co])

        if trace_ > 0 and det_ > 0 and trace_**2 > 4*det_:
            wave_type = 1
        elif trace_ > 0 and det_ > 0 and trace_**2 < 4*det_:
            wave_type = 2
        elif trace_ < 0 and det_ > 0 and trace_**2 > 4*det_:
            wave_type = 3
        elif trace_ < 0 and det_ > 0 and trace_**2 < 4*det_:
            wave_type = 4
        elif det_ < 0:
            wave_type = 5
        else:
            wave_type = 0

        singularities_df.loc[i, 'wave_type'] = wave_type

    return singularities_df

def wave_properites(singularities_df, wave_dict, sfreq, tol=1):
    """
    Find the durations of each wave, and the number of each wave type.

    Parameters
    ----------
    singularities_df : pd.DataFrame
        The singularities dataframe.
    wave_dict : dict
        The wave type dictionary.
    sfreq : int
        The sample rate.
    tol : int, optional
        The tolerance for spatial movement. The default is 2.

    Returns
    -------
    wprops_df : pd.DataFrame
        The wave properties dataframe.

    """

    wave_types = list(wave_dict.keys())

    wprops_df = pd.DataFrame(columns=['wave_type', 'number', 
                                      'mean_duration', 'std_duration', 'sum_duration'])

    durations = {wave_type: [] for wave_type in wave_types}

    for wave_type, group in singularities_df.groupby('wave_type'):

        # Find differences in space between each time point the wave is present
        x_diff = group['x_grid'].diff().abs()
        y_diff = group['y_grid'].diff().abs()
        # Find differences in space between each time point the wave is present
        time_diff = group['IA_frame'].diff().abs()
        time_diff.iloc[0] = 0
        # Determine if the wave has changed (either in space or has been absent for more than 1 second)
        pos_change = (x_diff > tol) | (y_diff > tol) | (time_diff > 1)
        pos_change.iloc[0] = True  # Always start with a change

        # Count the number of consecutive steps a wave is present in the same location (i.e. the duration of the wave)
        counts = pos_change.cumsum().value_counts().values
        durations[wave_type] = (counts / sfreq) * 1000  # Convert to ms

    for wave_type in wave_types:
        dur = durations.get(wave_type, [])
        wprops_df = wprops_df.append({
            'wave_type': wave_dict[wave_type],
            'number': len(dur),
            'mean_duration': np.mean(dur) if len(dur) > 0 else 0,
            'std_duration': np.std(dur) if len(dur) > 0 else 0,
            'sum_duration': np.sum(dur) if len(dur) > 0 else 0
        }, ignore_index=True)

    return wprops_df

# dicrete dynamics:

def wave_dominance(phi, singularities_df, ext=False,
                    standing_th=2, plane_th=0.85):
    """
    Find the dominant wave type at each timepoint
    -Dominance determined by the area of the singularity
    
    Parameters
    ----------
    phi : np.array
        The speed.
    singularities_df : pd.DataFrame
        The singularities dataframe.
    ext : bool, optional
        If should include standing and plane waves. The default is False.
    standing_th : float, optional
        The standing threshold (in std below mean). 
        The default is 2.
    plane_th : float, optional
        The plane threshold. The default is 0.85.

    Returns
    -------
    dom_waves : np.array
        The dominant wave type at each timepoint.

    """

    n_samples = len(phi)
    dom_waves = np.zeros(n_samples)
    for i in range(n_samples):
        if i in singularities_df['IA_frame'].values:
            if singularities_df[singularities_df['IA_frame']==i].shape[0] > 1:
                df = singularities_df[singularities_df['IA_frame']==i]
                dom_waves[i] = df.iloc[np.argmax(df['area'])]['wave_type']
            else:
                dom_waves[i] = singularities_df[singularities_df['IA_frame']==i]['wave_type'].values[0]
        else:
            dom_waves[i] = 0
        if ext:
            if phi[i] < (np.mean(phi)-standing_th*np.std(phi)):
                dom_waves[i] = 6
            elif phi[i] > plane_th:
                dom_waves[i] = 7
    return dom_waves

def wave_dom_persistence(dom_waves, wave_dict, sfreq):
    """
    Find the persistence of dominant wave types.

    Parameters
    ----------
    dom_waves : np.array
        The dominant wave type at each timepoint.
    wave_dict : dict
        The wave type dictionary.
    sfreq : int
        The sample rate.

    Returns
    -------
    persistence : dict
        The persistence of each wave type in ms.

    """

    wave_types = wave_dict.keys()
    persistence = {}
    for wave_type in wave_types:
        if wave_type not in dom_waves:
            persistence[wave_type] = 0
        else:
            count_list = []
            count = 0
            for i in range(len(dom_waves)):
                if dom_waves[i] == wave_type:
                    count += 1
                else:
                    if count > 0:
                        count_list.append(count)
                        count = 0
            persistence[wave_type] = count_list
    for k, v in persistence.items():
        persistence[k] = [x/sfreq*1000 for x in v] # convert to ms

    return persistence

def wave_dom_tpm(dom_waves, wave_dict):
    """
    Compute the transition probability matrix.

    Parameters
    ----------
    dom_waves : np.array
        The dominant wave type at each timepoint.
    wave_dict : dict
        The wave type dictionary.

    Returns
    -------
    tpm : np.array
        The transition probability matrix. (Past x Future)
        i.e y-axis is the current wave type, x-axis is the next wave type.

    """

    wave_types = wave_dict.keys()
    tpm = np.zeros((len(wave_types), len(wave_types)))
    for i in range(len(dom_waves)-1):
        tpm[int(dom_waves[i]), int(dom_waves[i+1])] += 1
    np.fill_diagonal(tpm, 0)
    tpm = tpm / tpm.sum(axis=1, keepdims=True)
    tpm[np.isnan(tpm)] = 0

    return tpm

def wave_dom_proportion(dom_waves, wave_dict):
    """
    Find the proportion of each wave type.

    Parameters
    ----------
    dom_waves : np.array
        The dominant wave type at each timepoint.
    wave_dict : dict
        The wave type dictionary.

    Returns
    -------
    proportions : dict
        The proportion of each wave type.

    """

    wave_types = wave_dict.keys()
    proportions = {}
    for wave_type in wave_types:
        proportions[wave_type] = np.sum(dom_waves==wave_type) / len(dom_waves)

    return proportions

def wave_dom_dynamics(dom_waves, wave_dict, sfreq):
    """
    Find the dynamics of the waves.

    Parameters
    ----------
    dom_waves : np.array
        The dominant wave type at each timepoint.
    wave_dict : dict
        The wave type dictionary.
    sfreq : int
        The sample rate.

    Returns
    -------
    dynamics : dict
        The dynamics of the waves.

    """

    persistence = wave_dom_persistence(dom_waves, wave_dict, sfreq)
    persistence = {k:np.mean(v) for k, v in persistence.items()}
    tpm = wave_dom_tpm(dom_waves, wave_dict)
    proportions = wave_dom_proportion(dom_waves, wave_dict)

    return proportions, persistence, tpm