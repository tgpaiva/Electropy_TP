from AutolabFuncs import *
from GamryFuncs import *

# Test functions

# CGD_files = import_gamry('/Users/tiagopaiva/Library/CloudStorage/OneDrive-UniversidadedeLisboa/CORKCAP_2022.05187.PTDC/Gamry1000EP6/T6_1/GCD/')

# data = parse_gcd(CGD_files)


# plot_gcd(data, saveplot = 'n', saveresults = 'n', outname = 'GCD')


CV_files = import_gamry('/Users/tiagopaiva/Library/CloudStorage/OneDrive-UniversidadedeLisboa/CORKCAP_2022.05187.PTDC/Gamry1000EP6/T6_1/CV/')# 
data = parse_cv(CV_files)
plot_cv(data)

# scanratelist = [item.scan_rate for item in data]
# np.round(np.array(scanratelist, dtype=float), 1)