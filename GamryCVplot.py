#!/usr/bin/env python
from GamryFuncs import *

## CV

def main(*args):

    arg = sys.argv[1]
    name =  sys.argv[2]

    CV_files = ImportGamry(arg)
    data = ParseCV(CV_files)
    PlotCV(data, saveplot = 'y', outname = name)

    return 

if __name__ == "__main__":
    import sys 
    main(*sys.argv)