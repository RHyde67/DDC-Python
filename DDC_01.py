# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 14:33:07 2018

@author: HydeR
Copyright R Hyde 2018
Released under the GNU GPLver3.0
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/
If you use this file please acknowledge the author and cite as a
reference:
Hyde, R.; Angelov, P., "Data density based clustering," Computational
Intelligence (UKCI), 2014 14th UK Workshop on , vol., no., pp.1,7, 8-10 Sept. 2014
doi: 10.1109/UKCI.2014.6930157
Downloadable from: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6930157&isnumber=6930143

NOTE: This is my first attempt at Python coding! Written with Spyder 3.3.2 using
    Python 3.6 and QT5 graphics backend (or plots don't come to foreground)
    Any constructive criticism, hints or tips are most welcome. If you are
    unsure of the intended functionality and are fanmiliar with Matlab, please
    see my original Matlab implementation.

Data Density Based Clustering with Manual Radii
Useage:
    CC, CR, Results = DDC01.DDC(DataIn, InitR, Verbose, Merge)
Inputs:
    DataIn:   m x n array of data for clustering m rows of samples, data
              should be normalised 0-1 or scaled appropriately
    InitR:    initial radii array with radius in each dimension, if a single
              value is provided the same value will be used for all.
    Verbose:  1 - plot output of progress, 0 - silent
    Merge:    Flag=1 to merge clusters if centre is within the ellipse of another
              NOTE: the merge implementation here produces different results from
              the Matlab version. I have not traced this discrepency yet. As a
              results, different initial radii are recommended.
Outputs:
    Results:  List of data contained in each cluster as arrays of each data dimension
        i.e. Results[0][0] are x-coords, Results[0][1] are y-coords etc
    ClusterCentres: list of arrays of cluster centre co-ords
    ClusterRadii: list of arrays of cluster radii
"""

# Function initialization
import numpy as np
import scipy.spatial as spatial
import matplotlib.pyplot as plt
from math import pi

# Algorithm Initialization
def DDC(DataIn, InitR, Verbose, Merge):
    # initialize
    if len(InitR) == 1: # check radii for each data dimension
        if Verbose == 1:
            print("Using equal radii")
        # InitR = [InitR]*DataIn.shape[1]
        InitR = np.ones([1, DataIn.shape[1]])*InitR
    elif len(InitR) == DataIn.shape[1]:
        InitR = np.atleast_2d( InitR )
    elif len(InitR) != DataIn.shape[1]:
        print("Number of Radii not 1 or equal to data dimensions")
        
    # Can only plot 2D
    if Verbose == 1 and DataIn.shape[1] != 2:
        Verbose = 0
        
    NumberOfClusters = 0; # no clusters at start
    Results = []
    ClusterRadii = []
    ClusterCentres = []
       
    
# DDC Algorithm Routine
    while DataIn.shape[0] > 0:
        # print(DataIn.shape) # bug trace for amount of remaining data
        NumberOfClusters += 1;
        ClusterRadii.append(InitR)
        
        ## Stage 1 initial cluster generation
        # Find initial data for cluster
        CentreIndex, Centre = find_centre(DataIn)
        Include = include_data(DataIn, CentreIndex, ClusterRadii[-1])
        # keep only those within 3*sigma
        Cluster = DataIn[np.array(Include[:,0])] # list of data in initial cluster
        Include = sigma_remove(Include, Cluster, Centre) # remove data outside 3-sigma
                
        ## STAGE 2 move cluster to local densest point and match radii to new clustered data
        # Move cluster centre to local densest point
        LocalCentre, LocalMean = find_centre(Cluster)
        LocalCentre = Include[LocalCentre] # convert local index to global index
        # assign data to new centre
        Include = include_data(DataIn, LocalCentre, ClusterRadii[-1])
        # remove outliers > 3*sigma
        Cluster = DataIn[np.array(Include[:,0])] # list of data in initial cluster
        Include = sigma_remove(Include, Cluster, Centre) # remove data outside 3-sigma
        # Update Radii to maximum distances on each axis
        ClusterRadii[-1] = update_radii(DataIn[Include], DataIn[LocalCentre], ClusterRadii[-1])
        
        ## Stage 3 reassign data to final cluster dimensions, and make final cluster adjustments
        # reassign data
        Include = include_data(DataIn, LocalCentre, ClusterRadii[-1])
        # remove outliers > 3*sigma
        Cluster = DataIn[np.array(Include[:,0])] # list of data in initial cluster
        Include = sigma_remove(Include, Cluster, Centre)
        # Update Radii to maximum distances on each axis
        ClusterRadii[-1] = update_radii(DataIn[Include], DataIn[LocalCentre], ClusterRadii[-1])
        
        ## Stage 4 save cluster information, clustered data and remove data from dataset
        # Note: cluster radii already saved in list
        ClusterCentres.append(DataIn[LocalCentre])
        # Results are list of data points and list of cluster assignment
        Results.append([ DataIn[Include][:,0,0], DataIn[Include][:,0,1] ])
        # remove data from initial dataset
        DataIn = np.delete(DataIn, Include, axis=0)
        
        if Verbose == 1:
            plot_remaining(DataIn)
            
    if Merge == 1:
            ClusterCentres, ClusterRadii, Results = merge_clusters(ClusterCentres, ClusterRadii, Results, Verbose)
        
    return ClusterCentres, ClusterRadii, Results


# Functions for DDC algorithm
def find_centre(Data): # finds index of densest point, i.e. cluster centre
    DataMean = np.atleast_2d(np.mean(Data,0))
    DataScalar = np.sum( np.square(Data) / Data.shape[0])
    DataDensity = spatial.distance.cdist(Data, DataMean, 'sqeuclidean') + DataScalar - np.sum(DataMean**2)
    CentreIndex = np.argmin(DataDensity)
    Centre = np.atleast_2d(Data[CentreIndex])
    return [CentreIndex, Centre]

def include_data(Data, CentreIndex, ClusterRadius) : # finds data within cluster
    Include = np.subtract(Data,Data[CentreIndex])**2 # square dim distances
    RadSq = (ClusterRadius)**2 # square each radius
    Include = np.sum(np.divide(Include,RadSq),1) # divide sqaure of dim distances by square of radii and add
    Include = np.argwhere(Include < 1)
    return Include
            
def sigma_remove(Include, Cluster, Centre): # removes data outside of 3*sigma
    InClusterDistances = spatial.distance.cdist(Cluster, Centre) # distances from centre
    Drop = np.argwhere( (InClusterDistances - np.mean(InClusterDistances)) > 3*np.std(InClusterDistances) )# list data outside of 3*sigma
    np.delete(Include, Drop) # remove indices >3*sigma
    return Include

def update_radii(Data, Centre, OldRadii):
    NewRadii = np.max(abs(Data - Centre), axis=0)
    for i in range(NewRadii.shape[1]):
        if NewRadii[0,i]<=0:
            NewRadii[0,i]=OldRadii[0,i]
    #NewRadii = np.ndarray.tolist(NewRadii)
    return NewRadii

# Function for simplisitic 'merge' of clusters where one cluster centre lies within another cluster
def merge_clusters(CC, CR, Results, Verbose): ## MUST BE BETTER WAY?? OK for single op, end of multiple ops?
    Merged = 1; # flag to indicate clusters have changed, so re-check merging
    
    while Merged == 1: # continue if merging has taken place
        Merged = 0
        Merges = np.zeros([np.shape(CC)[0],np.shape(CC)[0]]) # list of clusters to merge
        
        for idx1, TestCentre in enumerate(CC): # test which centres are in this cluster
            ## find centres within clusters ##
            Inside = ([(np.sum( (np.reshape(CC - TestCentre, [len(CC),2] ))**2 / CR[idx1]**2, axis=1)<1)*1]) # test centre in ellipse
            Merges[idx1,:] = Inside[0] # save list of clusters to merge in array of all cluster merges
            
        ## merge identified clusters ##
        np.fill_diagonal(Merges, 0)
        
        for idx1, CombineWith in enumerate(Merges):
            
            if any(CombineWith>0): # if any are to be combined
                Merged = 1 
                for idx2 in range(len(CombineWith)-1,-1,-1):
                    
                    if CombineWith[idx2] == 1:
                        
                        for idx3, unused in enumerate(Results[0]): # cycle through each data axis
                            
                            Results[idx1][idx3]=np.concatenate( ( Results[idx1][idx3], Results[idx2][idx3] ), axis=0)
                        
                        # create merged cluster info
                        CC[idx1] = np.atleast_2d( np.mean( Results[idx1], 1) ) # new centre is mean of merged data
                        for idx3, unused in enumerate(CC[idx1][0]):
                            CR[idx1][0][idx3] = max( abs( Results[idx1][idx3]-CC[idx1][0][idx3] ) )
                            
                # delete merged clusters
                for idx2 in range(len(CombineWith)-1,-1,-1):
                    
                    if CombineWith[idx2] == 1:
                        
                            del Results[idx2] # remove data from original cluster list
                            del CC[idx2] # delete merged cluster centre
                            del CR[idx2] # delete merged cluster radii
                            
                break
            
        if Verbose == 1:
            merge_plot(Results, CC, CR)          
                
   
    return [CC, CR, Results]


## Functions for visulization, i.e. 'Verbose = 1'
    # Function to plot remaining data during clustering
def plot_remaining(DataIn):
    plt.figure(1)
    plt.clf()
    plt.scatter(DataIn[:,0], DataIn[:,1], s=0.5, c='b')
    plt.show(block=False)
    plt.title("Remaining Data")
    plt.pause(0.05)
    
# Function for viualisation during merge function
def merge_plot(Results, CC, CR):
    plt.figure(99)
    plt.clf()
    plt.xlim([0,1])
    plt.ylim([0,1])
    for i in Results:
        plt.scatter(i[0], i[1], s=0.5)
    
    for i, C in enumerate(CC):
        R = CR[i]
        t = np.linspace(0, 2*pi, 100)
        plt.plot( C[0][0]+R[0][0]*np.cos(t) , C[0][1]+R[0][1]*np.sin(t) )
        plt.plot(C[0][0], C[0][1], marker='+', color='r')
        plt.text(C[0][0], C[0][1], i)
    plt.title("Merging Clusters\n(if Centre in another Cluster)")
    plt.pause(0.05)
    plt.show
    