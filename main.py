# -*- coding: utf-8 -*-
# noinspection PyUnresolvedReferences
"""
Created on Wed Dec  5 14:03:55 2018

@author: HydeR

Basic wrapper to load data and run my Implementation of Data Density Based Clustering:
R. Hyde and P. Angelov, “Data density based clustering,” in 2014 14th UK Workshop
    on Computational Intelligence (UKCI), 2014, pp. 1–7.

Written with Spyder 3.3.2 using Python 3.6 and QT5 graphics backend (or plots
    don't come to foreground)

Note: This is my first attempt at Python coding, any feedback is most welcome, especially constructive criticism!

Parameters:
    initial_radii: initial radii for clusters. DDC is robust to larger than necessary
            radii as they are adjusted by the algorithm. One radius will be set
            for all data dimensions, or provide one for each.
    verbose: set to 1 for in process plots and info. Note, using verbose will
            leave a plot of final clusters allowing you to differentiate between
            multiple clusters of similar colour but can be very slow.
    merge: set to perform a simple merge function whereby clusters with their
        centre inside another are combined

Example radii:
    for file DS2.csv, use 0.06
    for file Gaussian 5000.csv (~5000 data in each cluster) use 0.10
"""

# Initialise
import numpy as np
import DDC_01a as ddc
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# Close any open plot windows
plt.close('all')

# Load data
with open('Gaussian5000.csv') as f:
    lines = (line for line in f if not line.startswith('#'))
    data_in = np.loadtxt(lines, delimiter=',', skiprows=1)

data_in = data_in[:, 0:2:1]  # in this example, ground truth data is in the final column, not to be used for clustering!

# normalize each axis 0-1
# not required, but may improve accuracy and simplifies radii selection
data_in = (data_in-np.min(data_in, 0)) / (np.max(data_in, 0) - np.min(data_in, 0))

# set variables
initial_radii = (0.15, 0.06)  # cluster initial radii, use single radius for equal dimension radii, or 1 per each data dimension
verbose = 0  # flag to provide info and plots during cluster analysis (limited implementation)
merge = 1  # flag to do some basic cluster merging after 1st analysis (not currently implemented)

time_start = timer()
centre_list, radii_list, results = ddc.ddc(data_in, initial_radii, verbose, merge)
time_end = timer()
time_elapsed = time_end - time_start

# Display Results
plt.figure(98)
plt.clf()
plt.xlim([0, 1])
plt.ylim([0, 1])
for i in results:
    plt.scatter(i[0], i[1], s=0.5)
plt.title('{:d} Data Clustered in {:.4f} s' .format(data_in.shape[0], time_elapsed))
plt.show()


# Code to time DDC algorithm, e.g. 3 averages of 100 repetition
'''
# import timeit
setup = 
import numpy as np
import DDC_01 as DDC01
import matplotlib.pyplot as plt
with open('exampledata02.csv') as f:
    lines = (line for line in f if not line.startswith('#'))
    DataIn = np.loadtxt(lines, delimiter=',', skiprows=1)

# normalize each axis 0-1
DataIn = ( DataIn-np.min(DataIn,0) ) / (np.max(DataIn,0) - np.min(DataIn,0))

# set variables
InitR = [0.14] # cluster initial radii, use single radius for equal dimension radii, or 1 per each data dimension
Verbose = 0 # flag to provide info and plots during cluster analysis (limited implementation)
Merge = 0

R = 3
N = 1000
T = timeit.repeat(setup=setup, stmt = 'Results = DDC01.DDC(DataIn, InitR, Verbose, Merge)', repeat=R, number=N)
print ('Fastest time = ', "{:.3g}".format(min(T)/N), 'ms')
'''
