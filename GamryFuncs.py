import pandas as pd
import os
import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from natsort import natsorted
from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

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

class GamryLSV:
  def __init__(self, data):
    self.data = data

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
            plt.savefig(HOME_FOLDER / f"{kwargs['outname']}.pdf")  # default saves to the desktop
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
        outresults.to_excel(os.path.join(HOME_FOLDER, f'{outname}.xlsx'), index=False)

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
            plt.savefig(os.path.join(HOME_FOLDER , f"{kwargs['outname']}.pdf"))  # default saves to the desktop
        
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
            plt.savefig(HOME_FOLDER /  f"{kwargs['outname']}.pdf")  # default saves to the desktop
        
        except KeyError:   
            raise Exception('Please specify a figure output filename')
        

def parse_lsv(filename):
    """
    function to parse the data on LSV Gamry's DTA files 
    """
    CV_text_files = [list(datafile) for datafile in filename]
    exp_start = [_search_cv(txtfile) for txtfile in CV_text_files]

    sorted_data_list =[]

    for n in range(len(exp_start)):

        start = exp_start[n]
        data = CV_text_files[n]
        
        sorted_data = [[line.split('\t')[3], line.split('\t')[4], line.split('\t')[2]] for line in data[start[0]+3::]]
        sorted_data_list.append(GamryLSV(sorted_data))

    return sorted_data_list

#######
def plot_lsv(lsv_data, ax=None ,pH = 14, normalize_data = 'y' ,save_plot = 'n', color_ = 'black'):
    """
    plots LSV data 
    """
    fifth_LSV_cycle = pd.DataFrame(lsv_data[4].data, columns = ['Potential', 'Current', 'Time'],   dtype = float)

    if normalize_data == 'y':

        # Voltage vs RHE

        potential = fifth_LSV_cycle['Potential'] + 0.058 * pH + 0.204

        # current to current density 2 cm2 electrode

        current = fifth_LSV_cycle['Current'] / 2
        
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(potential, current, color=color_)    
    # fig, ax = plt.subplots()   2
    # ax.plot(potential, current, color = color_)

    # Axis cosmetics

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, -3))  # Force 10⁻³ scaling
    ax.yaxis.set_major_formatter(formatter)

    # ax.ticklabel_format(axis= 'y', style='scientific', scilimits=(-3,-3))

    ax.set_xlabel('Potential vs RHE (V)', weight = 'bold')
    ax.set_ylabel('Current Density (A/g)', weight = 'bold')

    # ax.legend(labels, frameon = False, loc = 'lower right', ncols = 2)
    # fig.tight_layout()

    # Save plot

    if save_plot == 'y':

        try:
            plt.savefig(HOME_FOLDER / f"{kwargs['outname']}.pdf")  # default saves to the desktop

        except KeyError:   
            raise Exception('Please specify a figure output filename')
        

    # calculate overpotential value at 10 mA/cm-2

    target = 0.010 # 10 mA/cm-2
    nearest_index = int((current - target).abs().idxmin())
    nearest_value = current.loc[nearest_index]
    overpotential = potential.tolist()[nearest_index]

    return overpotential


def plot_tafel(lsv_data, reaction_kind = 'OER', pH = 14, normalize_data = 'y' ,save_plot = 'n'):
    """
    plots LSV data 
    """

    fifth_LSV_cycle = pd.DataFrame(lsv_data[4].data, columns = ['Potential', 'Current', 'Time'],   dtype = float)

    if normalize_data == 'y':

        # Voltage vs RHE

        potential = fifth_LSV_cycle['Potential'] + 0.058 * pH + 0.204

        # current to current density 2 cm2 electrode

        if reaction_kind == 'HER':
            current = np.log10(fifth_LSV_cycle['Current'] / 2 * 1000 * -1)

        else:
            current = np.log10(fifth_LSV_cycle['Current'] / 2 * 1000) 


    fig, ax = plt.subplots()   
    ax.plot(current, potential, color = 'black')

    # Axis cosmetics

    ax.set_ylabel('Potential vs RHE (V)', weight = 'bold')
    ax.set_xlabel('Log Current Density (mA/g)', weight = 'bold')
    ax.set_xscale('log')
    # ax.legend(labels, frameon = False, loc = 'lower right', ncols = 2)
    fig.tight_layout()

    # Save plot

    if save_plot == 'y':

        try:
            plt.savefig(HOME_FOLDER / f"{kwargs['outname']}.pdf")  # default saves to the desktop

        except KeyError:   
            raise Exception('Please specify a figure output filename')
        

    return 

# def calculate_tafel_slope(lsv_data,  pH = 14, reaction_kind = 'OER'):
    
#     fifth_LSV_cycle = pd.DataFrame(lsv_data[4].data, columns = ['Potential', 'Current', 'Time'],   dtype = float)

#     if reaction_kind == 'HER':

#         X_full = fifth_LSV_cycle['Current'] / 2 * 1000 * -1
#         start = int(np.where(X_full < 1)[0][-1])
#         X_filtered = np.array(np.log10(X_full[start + 1:]))
#         print(X_filtered)

#     else:
        
#         X_full = (fifth_LSV_cycle['Current'] / 2 * 1000)
#         start = int(np.where(X_full < 1)[0][-1])
#         X_filtered = np.array(np.log10(X_full[start + 1:]))

#     y_full = np.array((fifth_LSV_cycle['Potential'] + 0.058 * pH + 0.204))

#     # discard nagative current density values

#     y_filtered = np.array(y_full[start + 1:])
#     best_mse = float('inf')
#     best_bvalue = float('inf')
#     best_model = None
#     best_chunk_index = -1
#     best_x, best_y, best_y_pred = None, None, None

#     # Split into 25 chunks (as evenly as possible)

#     step = len(X_filtered) / 50 # step to slice the data in 50 points at a time

#     x_chunks = np.array_split(X_filtered, step)
#     y_chunks = np.array_split(y_filtered, step)

#     # Process in 100-point chunks

#     for i, (x_chunk, y_chunk) in enumerate(zip(x_chunks, y_chunks)):

#         print(f"Chunk {i+1}: {len(x_chunk)} points")

#         # Reshape x for sklearn (must be 2D)
#         x_chunk_reshaped = x_chunk.reshape(-1, 1)

#         model = LinearRegression()
#         model.fit(x_chunk_reshaped, y_chunk)
#         y_pred = model.predict(x_chunk_reshaped)
        
#         mse = mean_squared_error(y_chunk, y_pred)
#         b_value = model.coef_

#         print(f"Chunk {i+1}: MSE = {mse:.8f}, Intercept = {model.intercept_:.4f}, Slope = {model.coef_[0]*1e3:.4f}")

#         if reaction_kind =='OER':

#             if mse < best_mse and b_value < best_bvalue and b_value > 0:

#                 best_mse = mse
#                 best_bvalue = b_value
#                 best_model = model
#                 best_chunk_index = i
#                 best_x, best_y, best_y_pred = x_chunk, y_chunk, y_pred

#         elif reaction_kind =='HER':

#             b_value = model.coef_ * -1
            
#             if mse < best_mse and b_value < best_bvalue and b_value > 0:

#                 best_mse = mse
#                 best_bvalue = b_value
#                 best_model = model
#                 best_chunk_index = i
#                 best_x, best_y, best_y_pred = x_chunk, y_chunk, y_pred

#     print(f"Best Slope: {best_model.coef_[0]*1e3:.4f} mV/Dec") # 1e3 conversion to mV
#     print(f"\nBest Model: Chunk {best_chunk_index+1} with MSE = {best_mse:.8f}")

#     fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)


#     axes[0].plot(best_x, best_y_pred, color='#00008B', linewidth = 10, alpha = 0.5, label='Best Fit Line')
#     axes[0].scatter(np.log10(X_full), y_full, color = 'black', s = 3)
#     axes[0].set_xscale('log')

#     axes[0].set_xlabel('Log J ($\\mathbf{mA~cm^{-2}})$', weight = 'bold')
#     axes[0].set_ylabel('Potential vs RHE (V)', weight = 'bold')

        
#     axes[1].scatter(best_x, best_y, color='black', label='Best Chunk Data')
#     axes[1].plot(best_x, best_y_pred, color='red', linewidth=2, label='Best Fit Line')
#     axes[1].set_xlabel('Log J ($\\mathbf{mA~cm^{-2}})$', weight = 'bold')
#     axes[1].set_ylabel('Potential vs RHE (V)', weight = 'bold')

#     return 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def calculate_tafel_slope(lsv_data, pH=14, reaction_kind='OER'):
    fifth_LSV_cycle = pd.DataFrame(lsv_data[4].data, columns=['Potential', 'Current', 'Time'], dtype=float)

    if reaction_kind == 'HER':
        X_full = fifth_LSV_cycle['Current'] / 2 * 1000 * -1 # flip y axis for HER
    else:
        X_full = fifth_LSV_cycle['Current'] / 2 * 1000

    start = int(np.where(X_full < 1)[0][-1])
    X_filtered = np.array(np.log10(X_full[start + 1:]))
    y_full = np.array(fifth_LSV_cycle['Potential'] + 0.058 * pH + 0.204)
    y_filtered = np.array(y_full[start + 1:])

    best_r2 = -np.inf
    best_bvalue = float('inf')
    best_model = None
    best_chunk_index = -1
    best_x, best_y, best_y_pred = None, None, None

    # Split data into chunks of ~50 points
    chunk_size = 50
    x_chunks = [X_filtered[i:i + chunk_size] for i in range(0, len(X_filtered), chunk_size)]
    y_chunks = [y_filtered[i:i + chunk_size] for i in range(0, len(y_filtered), chunk_size)]

    for i, (x_chunk, y_chunk) in enumerate(zip(x_chunks, y_chunks)):
        if len(x_chunk) < 2:
            continue  # Skip too small chunks

        print(f"Chunk {i + 1}: {len(x_chunk)} points")

        x_chunk_reshaped = x_chunk.reshape(-1, 1)
        model = LinearRegression()
        model.fit(x_chunk_reshaped, y_chunk)
        y_pred = model.predict(x_chunk_reshaped)

        r2 = r2_score(y_chunk, y_pred)
        b_value = model.coef_[0]

        print(f"Chunk {i + 1}: R² = {r2:.8f}, Intercept = {model.intercept_:.4f}, Slope = {b_value * 1e3:.4f}")

        if reaction_kind == 'OER':
            
            if r2 > best_r2 and 0 < b_value < best_bvalue:
                best_r2 = r2
                best_bvalue = b_value
                best_model = model
                best_chunk_index = i
                best_x, best_y, best_y_pred = x_chunk, y_chunk, y_pred

        elif reaction_kind == 'HER':
            b_value = -b_value  # Flip sign for HER

            if r2 > best_r2 and 0 < b_value < best_bvalue:
                best_r2 = r2
                best_bvalue = b_value
                best_model = model
                best_chunk_index = i
                best_x, best_y, best_y_pred = x_chunk, y_chunk, y_pred

    if best_model:
        print(f"\nBest Slope: {best_model.coef_[0]*1e3:.4f} mV/Dec")
        print(f"Best Model: Chunk {best_chunk_index + 1} with R² = {best_r2:.8f}")
    else:
        print("No valid model found.")

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    axes[0].plot(best_x, best_y_pred, color='#00008B', linewidth=3, alpha=0.5, label='Best Fit Line')
    axes[0].scatter(np.log10(X_full), y_full, color='black', s=3)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Log J ($\\mathbf{mA~cm^{-2}})$', weight='bold')
    axes[0].set_ylabel('Potential vs RHE (V)', weight='bold')

    axes[1].scatter(best_x, best_y, color='black', label='Best Chunk Data')
    axes[1].plot(best_x, best_y_pred, color='red', linewidth=2, label='Best Fit Line')
    axes[1].set_xlabel('Log J ($\\mathbf{mA~cm^{-2}})$', weight='bold')
    axes[1].set_ylabel('Potential vs RHE (V)', weight='bold')

    return
