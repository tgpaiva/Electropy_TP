import pandas as pd
import os
import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from natsort import natsorted

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

plt.rcParams['svg.fonttype'] = 'none' # none ; path ; svgfont (only for Chrome/Opera)

# Class definition

class GamryGCD:
  def __init__(self, index, data, current):
    self.index = index
    self.data = data
    self.current = current

class GamryCV:
  def __init__(self, data, scan_rate):
    self.data = data
    self.scan_rate = scan_rate

# DTA parsing

def _read_file(pathtofile):
    """
    Text file read generator
    """
    with open(pathtofile) as file:
        while lines := file.readline():
            yield lines

def _search_gcd_curve(textfile):
    """
    Search beginning of a data set
    """
    for idx, line in enumerate(textfile):
        if re.findall(r"Curve", line, re.IGNORECASE):
            return idx    

def _search_cv(textfile):
    """
    Search beginning of a data set
    """
    index_list = []

    for idx, line in enumerate(textfile):
        if re.findall(r"Curve", line, re.IGNORECASE):
            index_list.append(idx) 

    return index_list        


def _current_value(textfile):
    """
    Search beginning of a data set
    """
    for idx, line in enumerate(textfile):
        if re.search(r"ISTEP1", line, re.IGNORECASE):
            return line.split('	')[-2]   

def _cv_scan_rate(textfile):
    """
    Search beginning of a data set
    """
    for idx, line in enumerate(textfile):
        if re.search(r"SCANRATE", line, re.IGNORECASE):
            return line.split('	')[-2]   

# functions

def import_gamry(pathtofiles):
    """
    Gets list of filesnames with *.DTA extension
    """
    files = os.listdir(pathtofiles)
    files = natsorted(files)
    files_txt = [i for i in files if i.endswith('.DTA')]
    text_file = [_read_file(pathtofiles / txtfile) for txtfile in files_txt]

    return text_file


def parse_cv(filename):
    """
    Parse Gamry CV  from DTA
    """
    CV_text_files = [list(datafile) for datafile in filename]
    exp_start = [_search_cv(txtfile) for txtfile in CV_text_files]
    Scan_Rate = [_cv_scan_rate(txtfile) for txtfile in CV_text_files]
    
    sorted_data_list =[]

    for n in range(len(exp_start)):

        start = exp_start[n]
        data = CV_text_files[n]
        sorted_data = [[line.split('\t')[3], line.split('\t')[4],line.split('\t')[2]] for line in data[start[2] + 3: start[2] + start[2+1] - start[2]]]
        sorted_data_list.append(GamryCV(sorted_data, Scan_Rate[n]))

    return sorted_data_list


def parse_gcd(filename: list, start: int = 2, step: int = 3) -> list:
    """
    Parse Gamry CV data from a DTA file.
    """
    # Extract target text files based on start and step
    cv_text_files = [list(filename[i]) for i in range(start, len(filename), step)]

    # Parse experiment start lines and current values
    exp_starts = [_search_gcd_curve(txt) for txt in cv_text_files]
    currents = [_current_value(txt) for txt in cv_text_files]

    parsed_data = []

    for i, exp_start in enumerate(exp_starts):

        if exp_start is None:
            continue  # Skip this entry if exp_start is invalid

        skip_header = exp_start + 3
        data = cv_text_files[i][skip_header:]
        
        # Extract relevant columns from each data line
        sorted_data = [line.split('\t')[2:4] for line in data]
        parsed_data.append(GamryGCD(i, sorted_data, currents[i]))

    return parsed_data


def plot_cv(DatatoPlot, normalize = False, active_mass = 0.100E-3, colors = None, saveplot = 'y', labels = None, **kwargs):
    """
    Plots the CV data
    To  normalize current to density current supply an measurement for the active mass. Default 0.1 mg
    """    

    CV_3rdcurve = [pd.DataFrame(CVcurve.data, columns = ['Potential', 'Current', 'Time'], dtype = float) for CVcurve in DatatoPlot]

    if normalize == True:

        for n in range(len(CV_3rdcurve)):

            CV_3rdcurve[n]['Current'] = CV_3rdcurve[n]['Current'] / active_mass

    # Prepare Plot 

    fig, ax = plt.subplots()
    if colors == None:
        colors = ['#0a1170','#cc9456','#7a2233','#be3b49', '#6c727a', '#6b9fe3']
    # colors = ["#1b4f72", "#b45f06", "#1d8348", "#922b21", "#5b2c6f"]

    if labels == None:
        labels = ['5 mV/s' , '10 mV/s', '20 mV/s', '50 mV/s', '100 mV/s' , '200 mV/s' ]

    # Actually plot

    for index, color in zip(range(len(CV_3rdcurve)), colors):
        ax.plot(CV_3rdcurve[index]['Potential'], CV_3rdcurve[index]['Current'] , color = color, linewidth = 1.8, marker = None)

    # Axis cosmetics

    ax.set_xlabel('Potential (V)', weight = 'bold')
    ax.set_ylabel('Current (A)', weight = 'bold')

    if normalize == True:
        ax.set_ylabel('Current Density (A/g)', weight = 'bold')

    ax.legend(labels, frameon = False, loc = 'lower right', ncols = 2)
    fig.tight_layout()

    # Save plot

    if saveplot == 'y':
        try:
            plt.savefig(HOME_FOLDER / 'Desktop' / f"{kwargs['outname']}.pdf")  # default saves to the desktop
        except KeyError:   
            raise Exception('Please specify a figure output filename')

    return 


    
def calculate_capacitance(dataset, current_values = 1, saveresults = 'n', **kwargs):
    """
    calculates the capacitance of GCD curves
    Default current value 1 A/g otherwise input a list of current density or integer value
    """
    data_list = [
        pd.DataFrame(GCDcurve.data, columns=['Time', 'Potential'], dtype=float)
        for GCDcurve in dataset
                ]

    if isinstance(current_values, int):
        current_values = [current_values] * len(data_list)

    results = []

    for df, current_value in zip(data_list, current_values):
        # Handle charge time selection robustly
        max_potential = df['Potential'].max()
        charge_time_vals = df.loc[df['Potential'] == max_potential, 'Time']
        
        charge_time = charge_time_vals.iloc[0]

        discharge_start_idx = (
            np.where(df['Potential'] == max_potential)[0][0] + 1
        )

        discharge_time = (
            df['Time'].iloc[-1] - df['Time'].iloc[discharge_start_idx]
        )

        max_potential = df['Potential'].iloc[discharge_start_idx]

        max_potential = max_potential              

        q_charge = charge_time * current_value / max_potential
        q_discharge = discharge_time * current_value / max_potential

        coulombic_efficiency = (discharge_time / charge_time) * 100
        energy_density_factor = max_potential ** 2 / 2
        energy_charge = energy_density_factor * q_charge
        energy_discharge = energy_density_factor * q_discharge

        results.append({
            'Q charge (F/g)': q_charge,
            'Q discharge (F/g)': q_discharge,
            'Coulombic efficiency (%)': coulombic_efficiency,
            'Charge Time': charge_time,
            'Discharge Time': discharge_time,
            'Energy density charge': energy_charge,
            'Energy density discharge': energy_discharge,
        })

    outresults = pd.DataFrame(results)

    if saveresults == 'y':
        outname = kwargs.get('outname')
        if not outname:
            raise Exception('Please specify an output filename for the Excel file.')
        outresults.to_excel(os.path.join(HOME_FOLDER, "Desktop", f'{outname}.xlsx'), index=False)

    return outresults


def plot_gcd(dataset, ylimits = [0, 0.85], colors = None, labels = None, saveplot = 'n', **kwargs):
    """
    Plot all GCD curves at different current values for same value 
    """
    data_list = [pd.DataFrame(GCDcurve.data, columns = ['Time', 'Potential'], dtype = float)  for GCDcurve in dataset]

    # Plot cosmetics 

    fig, ax = plt.subplots()
    
    if colors == None:
        colors = ['#0a1170','#cc9456','#7a2233','#be3b49', '#6c727a', '#6b9fe3']

    # labels = [item.current/mass for item in dataset]

    if labels == None:
        labels = ['0.5 A/g' , '1 A/g', '2 A/g', '4 A/g', '8 A/g' , '10 A/g' ]

    # actually plot

    for index, color in zip(range(len(data_list)), colors):
        ax.plot(data_list[index]['Time'], data_list[index]['Potential'], color = color, linestyle = '', markersize = 1, marker = 'o')
        
    # Axis decoration

    ax.set_xlabel('Time', weight = 'bold')
    ax.set_ylabel('Potential (V) vs SCE', weight = 'bold')
    ax.set_ylim(ylimits)
    ax.legend(labels, frameon = False, markerscale=6)
    fig.tight_layout()

    if saveplot == 'y':
        try:
            plt.savefig(os.path.join(HOME_FOLDER , "Desktop" ,f"{kwargs['outname']}.pdf"))  # default saves to the desktop
        except KeyError:   
            raise Exception('Please specify a figure output filename')

    return


def plot_cycling_test(metrics_dataframe, saveplot = 'n', **kwargs):
    """ 
    Plots the result of a GCD cycling test experiment
     """
    fig, ax = plt.subplots()

    ax.plot(metrics_dataframe['Q discharge (F/g)'], marker = 'o', color = 'black', markersize = 5, linestyle = '')
    ax.set_xlabel('Cycles (n)', weight='bold')
    ax.set_ylabel('Capacity (F/g)', weight='bold')

    if saveplot == 'y':
        try:
            plt.savefig(HOME_FOLDER / "Desktop" / f"{kwargs['outname']}.pdf")  # default saves to the desktop
        except KeyError:   
            raise Exception('Please specify a figure output filename')