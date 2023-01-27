# hop-field

Proof of principle implementation of a 'tight binding field' model for molecular dynamics.  The model defines spatially resolved fields that encode the following:
1. Two-body atomic configuration energies
2. The parameters of a tight binding model, as a function of the distance between different types of atoms (initially C, H, and O).  This includes orbital energies, intraorbital hopping and electron-atom crystal field interactions. The symmetry of hopping and crystal field terms is defined by the Slater-Koster relations.

The software uses Monte Carlo to 

Limited development work has gone into testing the predictive power of a convolutional neural network (CNN) discriminator to emulate the model for specific molecules, and exploring alternatives to Monte Carlo for convergence (such as GFlowNet).

## Results





## Running the code

The model is initialized by running 'model_creation.py'.  Look within this file to define the modeling parameters.  Note that this part of the code is written in non-accelerated Python, and may take a few minutes if mid-size molecules (~100 atoms) are included.

Monte-carlo optimization of the model parameters is performed by running 'data_creation.py', which is numba accelerated and closely resembles our expected performance on a personal computer CPU.  The mc_data object created in 'data_creation.py' can be used to examine the convergence and as a source of training data for AI-assisted optimization.

