#!/usr/bin/env python

from GamryFuncs import *

## GCD

def main(*args):

    arg = sys.argv[1]
    name =  sys.argv[2]

    GCD_files = import_gamry(arg)
    data = parse_gcd(GCD_files)
    plot_gcd(data, saveplot = 'y' , saveresults = 'y', outname = name)

    return 

if __name__ == "__main__":
    import sys 
    main(*sys.argv)
