import os 
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from natsort import natsorted
from scipy.integrate import simpson, trapezoid

HOME_FOLDER = Path.home()
cwd = Path.cwd()
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
plt.rcParams['svg.fonttype'] = 'none' # none ; path ; svgfont (only for Chrome/Opera)


def import_data(path):
    """
    Import data and define outname
    """
    files = os.listdir(path)
    files_txt = [i for i in files if i.endswith('.txt')]
    files_txt = natsorted(files_txt)
    data = [pd.read_csv(path / dataset, delimiter = '\t', header = 0) for dataset in files_txt]
    name = re.split('/', str(path))[-1]

    return name, data

def parse_deposition_file(file_path):
    """
    import deposition data from autolab potentiostats
    returns a dataframe with time and potential columns
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        headers = None  # To store the initial header
        data = []       # To store the data rows
        for line in file:
            line = line.strip() # Strip whitespace and skip empty lines
            if not line:
                continue

            if line.startswith("Time (s)	WE(1).Potential (V)"):
                if headers is None:  # If headers are already set, skip repeated header lines
                    headers = line  # Store the first occurrence of the header
                continue  # Skip any subsequent headers

            data.append(line.split('\t'))
    data = pd.DataFrame(data[1:], columns=['Time', 'Potential']).apply(pd.to_numeric)

    return data

def plot_pulsed_deposit_data(data):
    """
    Plots deposition data
    """
    plt.plot(data['Time'], data['Potential'], color = 'black', 
             marker = 'o', linestyle = ' ', markersize = 5)
    plt.xlabel('Time (s)')
    plt.ylabel('Potential (V)')

def _order_cv_data(data):
    """
    Return dictionary with CV data arranged in a Python Dic with scan rates as the key
    """
    indeces  = [data[n].loc[(data[n] == 0).any(axis = 1)].index[-2]  for n in range(len(data))   ]
    cv_3rd_curve = [data[n].iloc[indeces[n]:]  for n in range(len(data))]
    labels = ['5 mV/s' , '10 mV/s', '20 mV/s', '50 mV/s', '100 mV/s' , '200 mV/s' ]
    cv_data = dict(zip(labels, cv_3rd_curve))

    return cv_data

def calculate_capacitance(dataset, save_results = 'n', ohmic_drop  = True, current_values = None, outname = 'GCD',):
    """ 
    Calculate performance metrics from GCD curves 
    Current_values if none are specified the default is 0.5, 1, 2, 4, 8, 10 A/g
    """
    df = [pd.concat([dataset[i], dataset[18 + i]], ignore_index = True) for i in range(2, 18, 3)]
    df = [pd.concat([frame['Time (s)'] - frame['Time (s)'].iloc[0], frame['WE(1).Potential (V)']], axis=1) for frame in df]

    if current_values is None:
        current_values = [0.5, 1, 2, 4, 8, 10]

    results = []
    for frame, current in zip(df, current_values):
        charge_time = frame['Time (s)'][frame['WE(1).Potential (V)'].idxmax()]

        if ohmic_drop is False:
            discharge_time = frame['Time (s)'].iloc[-1] - charge_time
            max_potential = frame['WE(1).Potential (V)'].max()

        elif ohmic_drop is True:
            discharge_time = frame['Time (s)'].iloc[-1] - frame['Time (s)'][int(frame['WE(1).Potential (V)'].idxmax()) + 1]
            max_potential = frame['WE(1).Potential (V)'][int(frame['WE(1).Potential (V)'].idxmax()) + 1]

        q_charge = charge_time * current / max_potential
        q_discharge = discharge_time * current / max_potential
        results.append({
            'Current': current,
            'Q charge (F/g)': q_charge,
            'Q discharge (F/g)': q_discharge,
            'Coulombic efficiency (%)': (discharge_time / charge_time) * 100,
            'Charge Time (s)': charge_time,
            'Discharge Time (s)': discharge_time,
            'Energy density charge': (frame['WE(1).Potential (V)'].max() ** 2 * q_charge) / 2,
            'Energy density discharge': (frame['WE(1).Potential (V)'].max() ** 2 * q_discharge) / 2,
        })

    if save_results == 'y':
        pd.DataFrame(results).to_excel(HOME_FOLDER / f'{outname}.xlsx', index=False)

def plot_gcd_curve(dataset, ylim = None ,save_plot = 'n', outname = 'GCD', labels = None):
    """
    Plot GCD curves at different current values and optionally save the plot and results.
    """
    # Prepare datasets

    df = [pd.concat([dataset[i], dataset[18 + i]]) for i in range(2, 18, 3)]
    df = [pd.concat([frame['Time (s)'] - frame['Time (s)'].iloc[0], frame['WE(1).Potential (V)']], axis=1) for frame in df]
    fig, ax = plt.subplots()
    colors = ['#0a1170', '#cc9456', '#7a2233', '#be3b49', '#6c727a', '#6b9fe3']
    
    if labels is None:
        labels = ['5 mV/s', '10 mV/s', '20 mV/s', '50 mV/s', '100 mV/s', '200 mV/s', '300 mV/s', '400 mV/s', '500 mV/s']

    if ylim is None:
        ylim = [-0.1, 0.9]

    for frame, color in zip(df, colors):
        ax.plot(frame['Time (s)'], frame['WE(1).Potential (V)'], color=color, linestyle = '', marker = 'o', markersize = 1)

    ax.set_xlabel('Time (s)', weight = 'bold')
    ax.set_ylabel('Potential (V) vs SCE', weight = 'bold')
    ax.legend(labels, frameon=False,  markerscale=6)
    ax.set_ylim(ylim)
    fig.tight_layout()

    if save_plot == 'y':
        plt.savefig(HOME_FOLDER / f'{outname}.pdf' )

def plot_cv(data, saveplot = 'n',  normalize = False, outname = 'CV', active_mass = None, colors = None ,labels = None):
    """
    Plot all CV curves at different scan rates for the same sample 
    """
    number_points = int(len(data[0])/3)
    cv_3rd_curve = [data[n][-number_points::] for n in range(len(data))]

    if normalize is True:
        for n, curve in enumerate(cv_3rd_curve):
            curve[n]['WE(1).Current (A)'] = curve[n]['WE(1).Current (A)'] / active_mass

    fig, ax = plt.subplots()
    if colors is None:
        colors = ['#0a1170','#cc9456','#7a2233','#be3b49', '#6c727a', '#6b9fe3']
    # colors = ["#1b4f72", "#b45f06", "#1d8348", "#922b21", "#5b2c6f"]

    if labels is None:
        labels = ['5 mV/s' , '10 mV/s', '20 mV/s', '50 mV/s', '100 mV/s' , '200 mV/s' ]

    for index, color in zip(range(len(cv_3rd_curve)), colors):
        ax.plot(cv_3rd_curve[index]['Potential applied (V)'], cv_3rd_curve[index]['WE(1).Current (A)'] , 
                color = color, linewidth = 1.8, marker = None)

    ax.set_xlabel('Potential (V)', weight = 'bold')
    ax.set_ylabel('Current (A)', weight = 'bold')

    if normalize is True:
        ax.set_ylabel('Current Density (A/g)', weight = 'bold')

    ax.legend(labels, frameon = False, loc = 'lower right', ncols = 2 )
    fig.tight_layout()

    if saveplot == 'y':
        plt.savefig(HOME_FOLDER / f'{outname}.pdf')

def integrate_cv(data, model = 'simpson'):
    """
    Integrate the cyclic voltammetry curve.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing 'Potential' and 'Current' columns.
        
    Returns:
        float: Integrated charge.
    """
    potential = data['Potential applied (V)'].values
    current = data['WE(1).Current (A)'].values

    if model == 'simpson':
        charge = simpson(current, x = potential) 

    elif model == 'trapz':
        charge = trapezoid(current, x = potential)

    elif model == 'paiva':
        # Calculate the differences in potential (ΔV)
        delta_v = np.diff(potential)
        # Calculate the average current values ((I_i + I_{i+1}) / 2)
        average_current = (current[:-1] + current[1:]) / 2
        # Calculate the area for each trapezoid (average_current * delta_v)
        areas = average_current * delta_v
        # Sum the areas to get the total integrated area
        charge = np.sum(areas)

    return charge


def plot_specific_cap(path): 
    """
    Plots specific capacitance for different charge/discharge currents
    """
    files = os.listdir(path)
    files = natsorted(files)
    data = [pd.read_excel(path + dataset) for dataset in files]
    fig, ax = plt.subplots()
    colors = ['#000000' , '#0a1170','#cc9456','#7a2233','#be3b49', '#6c727a', '#6b9fe3','#8386bd' ]
    labels = [filename.split('.')[-2] for filename in files]   

    for index, color in zip(range(len(data))  , colors):
        ax.plot(data[index]['Current'], data[index]['Q discharge (F/g)'], color = color, linewidth = 2, marker = 'o')

    ax.set_xlabel('Applied Current (A)', weight = 'bold')
    ax.set_ylabel('Specific Capacitance', weight = 'bold')
    ax.legend(labels, frameon = False)
    fig.tight_layout()

def plot_multiple_cv(pathtofiles, cv_scan_rate = '50 mV/s'):
    """
    Plots CV from different samples
    takes as first argument list of paths to files
    second argument is the CV scan rate, default is 50 mV/s
    """
    names = []
    cvdata = []

    for path in pathtofiles:
        name_out, cvdata_out = import_data(path)
        names.append(name_out)
        cvdata.append(cvdata_out)

    cv_data_dic = [_order_cv_data(data) for data in cvdata]
    fig, ax = plt.subplots()
    # colors = ['#0a1170','#cc9456','#7a2233','#be3b49', '#6c727a', '#6b9fe3']
    colors = [
    "#8B0000",  # Dark Red
    "#00008B",  # Dark Blue
    "#006400",  # Dark Green
    "#8B008B",  # Dark Magenta
    "#B8860B",  # Dark Goldenrod
    "#483D8B",  # Dark Slate Blue
    "#2F4F4F",  # Dark Slate Gray
    "#8B4513"   # Saddle Brown
    ]

    for index, data in enumerate(cv_data_dic):
        ax.plot(data[cv_scan_rate]['Potential applied (V)'], data[cv_scan_rate]['WE(1).Current (A)'], 
                label = names[index], color = colors[index])
        ax.set_xlabel('Potential (V)', weight = 'bold')
        ax.set_ylabel('Current (A)', weight = 'bold')
        ax.legend(frameon = False)
