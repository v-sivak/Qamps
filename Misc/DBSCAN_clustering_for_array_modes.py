a#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 13:46:42 2018

@author: Vladimir Sivak
"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib as mpl
from scipy.optimize import curve_fit, brentq
from devices_functions import *
import h5py




# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------




#device = JTLA02Test the 4-12 circulator at room T and 77 K
Scheduled: Oct 1, 2018 at 14:45 to 15:45

#device_folder = device.device_folder


dic = np.load(r'/Users/vs362/Downloads/KC180702_1_forVlad.npz')

currents = dic['currents']
ag_freqs = dic['frequencies']
data = dic['data']
z = data / np.abs(data)
for ind,I in enumerate(currents):
    z0 = z[ind][0]
    z[ind] = z[ind]/z0
z = np.angle(z,deg=True)


dpi = 150
fig1, ax1 = plt.subplots(1,dpi=dpi)
p1 = ax1.pcolormesh(currents*1e3, ag_freqs*1e-9, np.transpose(z), cmap='RdBu')
fig1.colorbar(p1,ax=ax1, label=r'$\rm Phase \,(deg)$')
ax1.set_ylabel(r'${\rm Frequency \,(GHz)}$')
ax1.set_xlabel(r'${\rm Current\,(mA)}$')



#hdf5 = h5py.File(device.data_file)
#try:
#    z = np.asarray(hdf5['two_tone_spectroscopy_data'].get('phase_normalized'))
#    ag_freqs = np.asarray(hdf5['two_tone_spectroscopy_data'].get('ag_freqs'))
#    currents = np.asarray(hdf5['two_tone_spectroscopy_data'].get('currents'))
#finally:
#    hdf5.close()


#device.plot_two_tone_spectroscopy()


# make a binary representation by threshholding the phase
threshhold = 11 #90 #55
bin_rep = np.empty_like(z)
for i, phase_arr in enumerate(z):
    for j, phase in enumerate(phase_arr):
        bin_rep[i][j] = 0 if np.abs(phase)<threshhold else 1

# deleting the noisy data at high frequency
cutoff = -1 #301 #700
ag_freqs = ag_freqs[:cutoff]
bin_rep_truncated = np.transpose(np.transpose(bin_rep)[:cutoff][:])

# colorplot the binary representation in black & white
fig2, ax2 = plt.subplots(1,dpi=150)
ax2.pcolormesh(currents*1e3, ag_freqs*1e-9, np.transpose(bin_rep_truncated), cmap='Greys')
ax2.set_ylabel(r'${\rm Frequency \,(GHz)}$')
ax2.set_xlabel(r'${\rm Current\,(mA)}$')
#
#
# convert the binary rep to the list of points on the 2D grid to which the clustering
# algorithm will be applied
points = []
for i, bit_arr in enumerate(bin_rep_truncated):
    for j, bit in enumerate(bit_arr):
        if bit==1: points.append( [i,j] )
        
# apply DBSCAN to separate points into clusters, use equal distance along both axes        
db = DBSCAN(eps=3, min_samples=10).fit(points) 
labels = db.labels_ #-1 labels the noise, i.e. points that don't have min_samples neighbors in their eps-vicinity 
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
new_arr = np.ones(np.shape(bin_rep_truncated))*(-1)
for index, point in enumerate(points):
    # some clusters have to be merged by hand
#    if labels[index] in [4,5]: labels[index]=4
#    if labels[index] in [5,7,8]: labels[index]=5
#    if labels[index] in [6,9]: labels[index]=6
    new_arr[point[0]][point[1]] = labels[index]
#n_clusters = 5 #7 # is modified due to manual merging of some clusters
#
#
#
#   

# colorplot the clusters
fig3, ax3 = plt.subplots(1,dpi=150)
p3 = ax3.pcolormesh(currents*1e3, ag_freqs*1e-9, np.transpose(new_arr), cmap='tab20') #tab20
fig3.colorbar(p3,ax=ax3, label=r'$\rm Cluster $')
ax3.set_ylabel(r'${\rm Frequency \,(GHz)}$')
ax3.set_xlabel(r'${\rm Current\,(mA)}$')
#fig3.savefig(device_folder + 'phase_colormap_clusters.png',dpi=240)
#
#
## select the average within each cluster as a proxy for the resonant frequency,
## add clusters to the object array_modes()
#array = array_modes()
#for cluster_num in range(0,n_clusters):
#    currs = []
#    frqs = []
#    for i, I in enumerate(currents):
#        cluster_indices = [j for j, x in enumerate(new_arr[i]) if x == cluster_num]
#        if cluster_indices:
#            currs.append(I)
#            ind = int(np.round(np.mean(np.asarray(cluster_indices))))
#            frqs.append(ag_freqs[ind])
#    array.add_mode(cluster_num+1,currs,frqs) #cluster numbering goes from zero, but mode indices from one
#
##array.mode_num_arr=[1, 2, 3, 5, 4, 6, 7] # need to order the mode indices correctly
## TODO: Automate this
#
#
#
#
#device.plot_array_modes()
##device.fit_array_modes(1000/2.35, 1/2-1/2.35, 0.5, 15, 40e9, 0.1) # JTLA01
##device.fit_array_modes(1000/2.5,1/2-1.5/2.5,0,25,50e9,0.1)
##device.plot_array_modes(fits = True)