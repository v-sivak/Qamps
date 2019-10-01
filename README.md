# Qamps
Characterization of quantum-limited amplifiers

Python scripts used to analyze data for characterization of quantum-limited parametric amplifiers in the lab of Michel Devoret at Yale (Qulab) in 2017-2019. This research resulted in the following papers:

https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.10.054020

https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.11.054060

https://arxiv.org/abs/1909.08005

Each new amplifier that is measured in the dilution fridge is created as an object of Amplifier class (later: either SPA or JAMPA class). The data is saved in hdf5 format. 

Data acquisition is done primarily using various measurement classes of Agilent PNA-X N5242A. Scripts for some specialized measurement sweeps and PNAX driver for qrlab are contained in 'pnax_sweeps'.
