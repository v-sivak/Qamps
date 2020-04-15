# Qamps
Characterization of quantum-limited amplifiers.

This repo contains python scripts used to analyze data in a few projects on quantum-limited parametric amplifiers that I worked on during my PhD in the lab of Michel Devoret (Qulab) at Yale University in 2017-2019. This research resulted in the following papers:

##### Optimizing the Nonlinearity and Dissipation of a SNAIL Parametric Amplifier for Dynamic Range
https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.10.054020

##### Kerr-Free Three-Wave Mixing in Superconducting Quantum Circuits
https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.11.054060

##### Josephson Array-Mode Parametric Amplifier
https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.13.024014

Data acquisition is done primarily using various measurement classes of Agilent PNA-X N5242A. Scripts for some specialized measurement sweeps and PNAX driver for qrlab are contained in 'pnax_sweeps'.

Each new amplifier measured in the dilution fridge is created as an object of Amplifier class (later: either SPA or JAMPA class). The data is saved in hdf5 format.
