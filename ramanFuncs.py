#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os 
from pathlib import Path
from scipy.optimize import curve_fit
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter, find_peaks

HOME_FOLDER = Path.home()

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 20

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def import_raman(file_name):
    """
    import function to read Horiba exported txt Raman data
    """
    data = np.loadtxt(file_name, delimiter = '\t')
    wave_number = data[:,0]
    counts = data[:,1]
    return wave_number , counts

def plot_raman(wave_number , counts, color_ = 'black', pp = False):
    """
    Plots raman spectrum
    Pick peaks if pp = True, default is False
    """
    fig, ax = plt.subplots()
    ax.plot(wave_number, counts, color = color_)
    ax.set_xlabel(r'Raman shift $\bf(cm^{-1})$', weight = 'bold')
    ax.set_ylabel('Counts', weight = 'bold')

    if pp == True:

        peaks, _ = find_peaks(counts, height = 97, distance = 10)
        # plt.plot(wave_number[peaks], base_corr_smothed[peaks], "o", label = 'Peaks', color = 'blue')
        plt.tight_layout()

        for xp,yp in zip(wave_number[peaks], counts[peaks]):
            plt.text(xp * 1.05, yp * 0.95, str(int(xp)), fontsize=10, rotation = 90, color = '#00008B')

    return wave_number, counts

def polynomial_baseline_correction(wave_number, spectrum_intensity, deg=3, baseline_indices = None):
    """
    Perform polynomial baseline correction on a spectrum.

    Args:
        x (np.ndarray): Independent variable (e.g., wavelength).
        y (np.ndarray): Dependent variable (spectrum with baseline).
        deg (int, optional): Degree of the polynomial for baseline fitting. Defaults to 3.
        baseline_indices (np.ndarray, optional): Indices of the baseline regions. If None, use entire spectrum.

    Returns:
        tuple: (corrected_spectrum, fitted_baseline)
    """
    if baseline_indices is None:
        baseline_indices = np.arange(len(wave_number))
    
    # Select baseline regions
    x_baseline = wave_number[baseline_indices]
    y_baseline = spectrum_intensity[baseline_indices]

    # Fit a polynomial to the baseline regions
    coeffs = np.polyfit(x_baseline, y_baseline, deg)
    baseline_poly = np.poly1d(coeffs)
    
    # Evaluate the polynomial on the entire x range
    fitted_baseline = baseline_poly(wave_number)
    
    # Correct the spectrum
    corrected_spectrum = spectrum_intensity - fitted_baseline
    
    return corrected_spectrum, fitted_baseline

def asymmetric_least_squares_smoothing(spectrum_intensity, lam= 1e5, p = 1E-5, niter = 10):
    """
    Perform Asymmetric Least Squares Smoothing to correct the baseline of a signal.

    Parameters:
    y (numpy.ndarray): The input signal (1D array).
    lam (float): The smoothing parameter. Larger values make the baseline smoother.
    p (float): The asymmetry parameter. Must be between 0 and 1.
    niter (int): The number of iterations to perform.

    Returns:
    numpy.ndarray: The estimated baseline of the signal.
    """
    L = len(spectrum_intensity)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)
    
    for i in range(niter):
        W = sparse.diags(w, 0)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * spectrum_intensity)
        w = p * (spectrum_intensity > z) + (1 - p) * (spectrum_intensity < z)

    
    return z

def apply_Savitzky_Golay(spectrum_intensity): 
    """
    Apply the Savitzky-Golay filter
    """
    window_length = 21  # Choose an odd number greater than the polynomial order
    polyorder = 2       # Choose the order of the polynomial fit
    y_smooth = savgol_filter(spectrum_intensity, window_length, polyorder)

    return y_smooth


def main(*argv):
        file_name = argv[1]
        wave_number , counts = import_raman(file_name) 
        baseline_fit = asymmetric_least_squares_smoothing(counts)
        corrected_signal = counts - baseline_fit
        base_corr_smothed = apply_Savitzky_Golay(corrected_signal)
        plot_raman(wave_number, base_corr_smothed)
        plt.savefig(HOME_FOLDER / f"{kwargs['file_name']}.svg")


if __name__ == "__main__":
    import sys
    main(*sys.argv)
    if len(sys.argv) != 2:
        print("Usage: python script.py raman_filename")
        sys.exit(1)