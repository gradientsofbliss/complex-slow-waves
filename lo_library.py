"""
29.06.2024

Tools to characterise low-dimensional brain dynamics

@author: George Blackburne

"""

import numpy as np
import pandas as pd

def amp_env(data):
    """
    Compute the amplitude envelope

    Parameters
    ----------
    timeseries : array

    Returns
    -------
    env : array
    """

    from scipy.signal import hilbert

    # if 1D array
    if len(data.shape) == 1:
        env = np.abs(hilbert(data))

    # if 2D array
    elif len(data.shape) == 2:
        env = np.zeros(data.shape)
        for i in range(data.shape[0]):
            env[i,:] = np.abs(hilbert(data[i,:]))

    return env

def auto_mi(data, sfreq, option='minima', thresh=1, t_max=None):
    '''
    Stabilisation of Auto-Mutual Information

    Parameters
    ----------
    data : array
    sfreq : sampling frequency
    option : 'minima' (e.g 1) or 'percentile' (e.g 0.5)
    thresh : threshold for minima/maxima
    t_max : maximum time lag

    Returns
    -------
    timescale : float

    '''
    import numpy as np
    from scipy.stats import linregress

    from statsmodels.tsa.stattools import acf
    
    if t_max is None:
        n_lags = len(data) - 1

    else:
        n_lags = t_max

    ac = acf(data, nlags=n_lags)[1:]
    ac = -np.log(1-(ac**2))/2 

    if option == 'minima':

        ac_final = ac[-int(n_lags/3):]
        slope, intercept, r_value, p_value, std_err = linregress(np.arange(len(ac_final)), ac_final)

        zero_cross = np.where(ac < intercept+(thresh*np.std(ac)))[0][0]

    if option == 'percentile':

        half_max = np.max(ac)/(100/thresh)

        zero_cross = np.where(ac < half_max)[0][0]
    
    timescale = zero_cross/sfreq * 1000 # in ms

    return timescale

def max_lambda(timeseries, sfreq):
    '''
    Compute the maximum Lyapunov exponent of a timeseries

    Parameters
    ----------
    timeseries : array
        1D array of a timeseries

    Returns
    -------
    lyap : float
    '''

    import nolds

    lyap = nolds.lyap_r(timeseries, 
                        tau=(1/sfreq))

    return lyap

def pca_c(data, n_comp=10, method='raw'):
    '''
    Compute eigenvectors and eigenvalues
    via linear PCA

    Parameters
    ----------
    data : 2D array
        C x T (channels x time) array of multivariate timeseries

    Returns
    -------
    eigvals : 1D array
        The eigenvalues of the covariance matrix
    eigvecs : 2D array
        The eigenvectors of the covariance matrix
    '''

    # if data is in the wrong shape then print warning
    if data.shape[0] > data.shape[1]:
        print('Warning: you have more channels than time points, are you sure this is correct?')

    data = data.T

    if method == 'raw':
        data = data - np.mean(data, axis=0)
        cov = np.cov(data, rowvar=False)
        eigvals, eigvecs = np.linalg.eig(cov)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        eigvals = eigvals[:n_comp]
        eigvecs = eigvecs[:, :n_comp]
    if method == 'svd':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_comp)
        pca.fit(data)
        eigvecs = pca.components_
        eigvals = pca.explained_variance_

    return eigvecs, eigvals

def msd(t0, start, t_max, signals):
    '''
    Compute the mean squared displacement of the multivariate signals 
    for each time-lag t (from 1 to t_max) with respect to a reference time point t0.

    https://doi.org/10.1038/s41467-021-26268-x

    Parameters
    ----------
    t0 : int
        The reference time point.
    start : int
        The starting time-lag (in samples).
    t_max : int
        The maximum time-lag (in samples).
    signals : array (n_signals, n_samples)
        Multivariate signals.

    Returns
    -------
    msd : array (t_max-1,)
        The mean squared displacement for each time-lag t.

    '''
    
    msd = []
    for t in range(start, t_max):
        msd.append(np.mean(np.abs(signals[:,t0+t] - signals[:,t0])**2))
    
    return msd

def p_dist_gauss(dat, ds=np.arange(0,101), bandwidth=1):
    '''
    Calculate the probability distribution and energy for each dt 
    using the kernel density estimation (KDE) method.

    https://doi.org/10.1038/s41467-021-26268-x

    Parameters
    ----------
    dat : array (n_samples,)
        The data to be used for the kernel density estimation.
    ds : array (n_samples,)
        The data points for which the probability distribution and energy will be calculated.
    bandwidth : float
        The bandwidth for the kernel density estimation.

    Returns
    -------
    nrg : array (n_samples,)
        The energy for each data point in ds.

    '''

    from statsmodels.nonparametric.kernel_density import KDEMultivariate
    
    if bandwidth is None:
        bandwidth = 1

    kde = KDEMultivariate(dat, bw=[bandwidth], var_type='c')
    pdf = kde.pdf(ds)
    
    energy = -np.log(pdf)

    return energy


