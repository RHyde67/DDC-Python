# -*- coding: utf-8 -*-
"""
created on Wed Dec  5 14:33:07 2018
Tidied up, refactored and generally mode compliant formatting 01/11/22. No change to functionality.
Warnign remains: 'Type 'list' doesn't have expected attribute '__sub__'' in line 122??

@author: HydeR
Copyright R Hyde 2018
Released under the GNU GPL ver3.0
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/
If you use this file please acknowledge the author and cite as a
reference:
Hyde, R.; Angelov, P., "data density based clustering," Computational
Intelligence (UKCI), 2014 14th UK Workshop on , vol., no., pp.1,7, 8-10 Sept. 2014
doi: 10.1109/UKCI.2014.6930157
Downloadable from: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6930157&isnumber=6930143

NOTE: This is my first attempt at Python coding! Written with Spyder 3.3.2 using
    Python 3.6 and QT5 graphics backend (or plots don't come to foreground)
    Any constructive criticism, hints or tips are most welcome. If you are
    unsure of the intended functionality and are unfamiliar with Matlab, please
    see my original Matlab implementation.

data Density Based clustering with Manual Radii
Useage:
    cluster_centre, cluster_radius, results = DDC01.DDC(data_in, initial_radius, verbose, Merge)
Inputs:
    data_in:   m x n array of data for clustering m rows of samples, data
              should be normalised 0-1 or scaled appropriately
    initial_radius:    initial radii array with radius in each dimension, if a single
              value is provided the same value will be used for all.
    verbose:  1 - plot output of progress, 0 - silent
    Merge:    Flag=1 to merge clusters if centre is within the ellipse of another
              NOTE: the merge implementation here produces different results from
              the Matlab version. I have not traced this discrepancy yet. As a
              results, different initial radii are recommended.
Outputs:
    results:  List of data contained in each cluster as arrays of each data dimension
        i.e. results[0][0] are x-coords, results[0][1] are y-coords etc
    cluster_centres: list of arrays of cluster centre co-ords
    cluster_radii: list of arrays of cluster radii
"""

# Function initialization
import numpy as np
import scipy.spatial as spatial
import matplotlib.pyplot as plt
from math import pi


# Algorithm Initialization
def ddc(data_in, initial_radius, verbose, merge):
    # merge argument is for additional option to merge clusters that 'overlap' to be implemented later
    # initialize
    if len(initial_radius) == 1:  # check radii for each data dimension
        if verbose == 1:
            print("Using equal radii")
        # initial_radius = [initial_radius]*data_in.shape[1]
        initial_radius = np.ones([1, data_in.shape[1]]) * initial_radius
    elif len(initial_radius) == data_in.shape[1]:
        initial_radius = np.atleast_2d(initial_radius)
    elif len(initial_radius) != data_in.shape[1]:
        print("Number of Radii not 1 or equal to data dimensions")

    # Can only plot 2D
    if verbose == 1 and data_in.shape[1] != 2:
        verbose = 0

    number_of_clusters = 0  # no clusters at start
    results = []
    cluster_radii = []
    cluster_centres = []

    # DDC Algorithm Routine
    while data_in.shape[0] > 0:
        # print(data_in.shape) # bug trace for amount of remaining data
        number_of_clusters += 1
        cluster_radii.append(initial_radius)

        # Stage 1 initial cluster generation
        # Find initial data for cluster
        centre_index, centre = find_centre(data_in)
        include = include_data(data_in, centre_index, cluster_radii[-1])
        # keep only those within 3*sigma
        cluster = data_in[np.array(include[:, 0])]  # list of data in initial cluster
        include = sigma_remove(include, cluster, centre)  # remove data outside 3-sigma

        # STAGE 2 move cluster to local densest point and match radii to new clustered data
        # Move cluster centre to local densest point
        local_centre, local_mean = find_centre(cluster)
        local_centre = include[local_centre]  # convert local index to global index
        # assign data to new centre
        include = include_data(data_in, local_centre, cluster_radii[-1])
        # remove outliers > 3*sigma
        cluster = data_in[np.array(include[:, 0])]  # list of data in initial cluster
        include = sigma_remove(include, cluster, centre)  # remove data outside 3-sigma
        # Update Radii to maximum distances on each axis
        cluster_radii[-1] = update_radii(data_in[include], data_in[local_centre], cluster_radii[-1])

        # Stage 3 reassign data to final cluster dimensions, and make final cluster adjustments
        # reassign data
        include = include_data(data_in, local_centre, cluster_radii[-1])
        # remove outliers > 3*sigma
        cluster = data_in[np.array(include[:, 0])]  # list of data in initial cluster
        include = sigma_remove(include, cluster, centre)
        # Update Radii to maximum distances on each axis
        cluster_radii[-1] = update_radii(data_in[include], data_in[local_centre], cluster_radii[-1])

        # Stage 4 save cluster information, clustered data and remove data from dataset
        # Note: cluster radii already saved in list
        cluster_centres.append(data_in[local_centre])
        # results are list of data points and list of cluster assignment
        results.append([data_in[include][:, 0, 0], data_in[include][:, 0, 1]])
        # remove data from initial dataset
        data_in = np.delete(data_in, include, axis=0)

        if verbose == 1:
            plot_remaining(data_in)

    if merge == 1:
        cluster_centres, cluster_radii, results = merge_clusters(cluster_centres, cluster_radii, results, verbose)

    return cluster_centres, cluster_radii, results


# Functions for DDC algorithm
def find_centre(data):  # finds index of densest point, i.e. cluster centre
    data_mean = np.atleast_2d(np.mean(data, 0))
    data_scalar = np.sum(np.square(data) / data.shape[0])
    data_density = spatial.distance.cdist(data, data_mean, 'sqeuclidean') + data_scalar - np.sum(data_mean ** 2)
    centre_index = np.argmin(data_density)
    centre = np.atleast_2d(data[centre_index])
    return [centre_index, centre]


def include_data(data, centre_index, cluster_radius):  # finds data within cluster
    include = np.subtract(data, data[centre_index]) ** 2  # square dim distances
    radius_squared = cluster_radius ** 2  # square each radius
    include = np.sum(np.divide(include, radius_squared), 1)  # divide square of dim distances by square of radii and add
    include = np.argwhere(include < 1)
    return include


def sigma_remove(include, cluster, centre):  # removes data outside of 3*sigma
    in_cluster_distances = spatial.distance.cdist(cluster, centre)  # distances from centre
    # list data outside 3*sigma
    drop = np.argwhere((in_cluster_distances - np.mean(in_cluster_distances)) > 3 * np.std(in_cluster_distances))
    np.delete(include, drop)  # remove indices >3*sigma
    return include


def update_radii(data, centre, old_radii):
    new_radii = np.max(abs(data - centre), axis=0)
    for i in range(new_radii.shape[1]):
        if new_radii[0, i] <= 0:
            new_radii[0, i] = old_radii[0, i]
    # new_radii = np.ndarray.tolist(new_radii)
    return new_radii


# Function for simplistic 'merge' of clusters where one cluster centre lies within another cluster
def merge_clusters(cluster_centre, cluster_radius, results, verbose):  # MUST BE A BETTER WAY?? OK for single op, end of multiple ops?
    merged = 1  # flag to indicate clusters have changed, so re-check merging

    while merged == 1:  # continue if merging has taken place
        merged = 0
        merge = np.zeros([np.shape(cluster_centre)[0], np.shape(cluster_centre)[0]])  # list of clusters to merge

        for idx1, test_centre in enumerate(cluster_centre):  # test which centres are in this cluster
            # find centres within clusters #
            inside = ([(np.sum((np.reshape(cluster_centre - test_centre, [len(cluster_centre), 2])) ** 2 / cluster_radius[idx1] ** 2,
                               axis=1) < 1) * 1])  # test centre in ellipse
            merge[idx1, :] = inside[0]  # save list of clusters to merge in array of all cluster merge

        # merge identified clusters #
        np.fill_diagonal(merge, 0)

        for idx1, combine_with in enumerate(merge):

            if any(combine_with > 0):  # if any are to be combined
                merged = 1
                for idx2 in range(len(combine_with) - 1, -1, -1):

                    if combine_with[idx2] == 1:

                        for idx3, unused in enumerate(results[0]):  # cycle through each data axis

                            results[idx1][idx3] = np.concatenate((results[idx1][idx3], results[idx2][idx3]), axis=0)

                        # create merged cluster info
                        cluster_centre[idx1] = np.atleast_2d(np.mean(results[idx1], 1))  # new centre is mean of merged data
                        for idx3, unused in enumerate(cluster_centre[idx1][0]):
                            cluster_radius[idx1][0][idx3] = max(abs(results[idx1][idx3] - cluster_centre[idx1][0][idx3]))

                # delete merged clusters
                for idx2 in range(len(combine_with) - 1, -1, -1):

                    if combine_with[idx2] == 1:
                        del results[idx2]  # remove data from original cluster list
                        del cluster_centre[idx2]  # delete merged cluster centre
                        del cluster_radius[idx2]  # delete merged cluster radii

                break

        if verbose == 1:
            merge_plot(results, cluster_centre, cluster_radius)

    return [cluster_centre, cluster_radius, results]


# Functions for visualisation, i.e. 'verbose = 1'
# Function to plot remaining data during clustering
def plot_remaining(data_in):
    plt.figure(1)
    plt.clf()
    plt.scatter(data_in[:, 0], data_in[:, 1], s=0.5, c='b')
    plt.show(block=False)
    plt.title("Remaining data")
    plt.pause(0.05)


# Function for visualisation during merge function
def merge_plot(results, cluster_centre, cluster_radius):
    plt.figure(99)
    plt.clf()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    for i in results:
        plt.scatter(i[0], i[1], s=0.5)

    for i, c in enumerate(cluster_centre):
        radius = cluster_radius[i]
        t = np.linspace(0, 2 * pi, 100)
        plt.plot(c[0][0] + radius[0][0] * np.cos(t), c[0][1] + radius[0][1] * np.sin(t))
        plt.plot(c[0][0], c[0][1], marker='+', color='r')
        plt.text(c[0][0], c[0][1], i)
    plt.title("Merging clusters\n(if centre in another cluster)")
    plt.pause(0.05)
    plt.show()
