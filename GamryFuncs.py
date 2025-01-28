import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted

home_folder = os.path.expanduser('~')

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
    text_file = [_read_file(pathtofiles + txtfile) for txtfile in files_txt]

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


def parse_gcd(filename, start = 2, step = 3):
    """
    Parse Gamry CV from DTA
    """
    CV_text_files = [list(filename[n]) for n in range(start,len(filename), step)]

    exp_start = [_search_gcd_curve(txtfile) for txtfile in CV_text_files]

    current = [(_current_value(txtfile)) for txtfile in CV_text_files]
    sorted_data_list = []

    for n in range(len(exp_start)):

        start = exp_start[n]
        data = CV_text_files[n]
        
        sorted_data = [[line.split('\t')[2], line.split('\t')[3]] for line in data[start + 3::]]

        sorted_data_list.append(GamryGCD(start, sorted_data, current[n]))

    return sorted_data_list

def plot_cv(DatatoPlot, normalize = False, active_mass = None, colors = None, saveplot = 'y', labels = None, **kwargs):
    """
    Plots the CV data
    """

    CV_3rdcurve = [pd.DataFrame(CVcurve.data, columns = ['Potential', 'Current', 'Time'], dtype = float) for CVcurve in DatatoPlot]

# new ####
    if normalize == True:

        for n in range(len(CV_3rdcurve)):

            CV_3rdcurve[n]['Current'] = CV_3rdcurve[n]['Current'] / active_mass

# new ####

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
    # ax.set_ylim([-0.075, 0.075])

    # Save plot

    if saveplot == 'y':
        try:
            plt.savefig(home_folder + '/Desktop/' + kwargs['outname'] + '.pdf')  # default saves to the desktop
        except KeyError:   
            raise Exception('Please specify a figure output filename')

    return 
    

def calculate_capacitance(dataset, current_values = 1, saveresults = 'y', **kwargs):
    """
    calculates the capacitance of GCD curves
    Default current value 1 A/g otherwise input a list of current density or integer value
    """
    data_list = [pd.DataFrame(GCDcurve.data, columns = ['Time', 'Potential'], dtype = float)  for GCDcurve in dataset]
    
    # Initialize numpy arrays
    
    Capacitance_charge = np.array([])
    Capacitance_discharge = np.array([])
    Coloumbiceff = np.array([])
    charge_times = np.array([])
    discharge_times = np.array([])
    Energy_density_charge = np.array([])
    Energy_density_discharge = np.array([])

    # checks if current_values is a int or a list of current ints
    
    if isinstance(current_values, int) == True:
        current_values = np.repeat(current_values, len(data_list)) 

    for GCD_curve, current_value in zip(range(len(data_list)), current_values):

        # Extract charge and discharge times
        
        max_potential = data_list[GCD_curve]['Potential'].max()
        charge_time = np.array(data_list[GCD_curve]['Time'].iloc[
            np.where(data_list[GCD_curve]['Potential'] == max_potential)
        ])
        discharge_time = np.array(data_list[GCD_curve]['Time'].iloc[-1:]) - charge_time

        # Concatenate charge and discharge times
        
        charge_times = np.concatenate((charge_times, charge_time))
        discharge_times = np.concatenate((discharge_times, discharge_time))

        # Calculate charge and discharge capacities
        
        q_charge = charge_time * current_value / max_potential
        q_discharge = discharge_time * current_value / max_potential

        # Compute and store performance metrics
        
        Coloumbiceff = np.concatenate((Coloumbiceff, (discharge_time / charge_time) * 100))
        Capacitance_charge = np.concatenate((Capacitance_charge, q_charge))
        Capacitance_discharge = np.concatenate((Capacitance_discharge, q_discharge))
        
        energy_density_factor = max_potential ** 2 / 2
        Energy_density_charge = np.concatenate((Energy_density_charge, energy_density_factor * q_charge))
        Energy_density_discharge = np.concatenate((Energy_density_discharge, energy_density_factor * q_discharge))

 
    # Write the results dataframe

    outresults = pd.DataFrame({'Q charge (F/g)': Capacitance_charge, 'Q discharge (F/g)': Capacitance_discharge, 'Coulombic efficiency (%)': Coloumbiceff, 'Charge Time': charge_times, 'Discharge Time': discharge_times, 'Energe density charge' : Energy_density_charge, 'Energe density discharge' : Energy_density_discharge })
    
    if saveresults == 'y':
        try:
            outresults.to_excel(home_folder + '/Desktop/' + kwargs['outname'] + '.xlsx')
        except KeyError:   
            raise Exception('Please specify an output filename for the excel file')

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
        ax.plot(data_list[index]['Time'], data_list[index]['Potential'], color = color, linewidth = 2, marker = None)
        
    # Axis decoration

    ax.set_xlabel('Time', weight = 'bold')
    ax.set_ylabel('Potential (V) vs SCE', weight = 'bold')
    ax.set_ylim(ylimits)
    ax.legend(labels, frameon = False)
    fig.tight_layout()

    if saveplot == 'y':
        try:
            plt.savefig(home_folder + '/Desktop/' + kwargs['outname'] + '.pdf')  # default saves to the desktop
        except KeyError:   
            raise Exception('Please specify a figure output filename')

    return