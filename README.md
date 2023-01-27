# hop-field

Proof of principle implementation of a 'tight binding field' model for molecular dynamics.  The model defines spatially resolved fields that encode the following:
1. Two-body atomic configuration energies
2. The parameters of a tight binding model, as a function of the distance between different types of atoms (initially C, H, and O).  This includes orbital energies, intraorbital hopping and electron-atom crystal field interactions. The symmetry of hopping and crystal field terms is defined by the Slater-Koster relations.

The software uses Monte Carlo to optimize the field parameters based on experimentally determined moleculare structures loaded in "Molecular Structure Data". Limited development work has gone into testing the predictive power of a convolutional neural network (1D CNN) discriminator to emulate the model for specific molecules (see run_ai.py), and exploring alternatives to Monte Carlo for convergence (such as GFlowNet).

## Results

![Convergence with and without hopping fields](Convergence_no_hopping.png | width=20)

A quick overview of some demo results from the code can be found in the 'Summary' PDF.  Key takeaways are:
1. Bond length accuracy is comparable to density functional theory (DFT), and there is significant room for improvement.
2. Disabling the hopping terms dramatically reduces the accuracy of the model for test molecules that it has not been trained on.  In this scenario, the model can still parse distinctions between single and double bonds, but no longer has knowledge of quantum mechanics.
3. The Numba-optimized Monte Carlo loop runs far faster than DFT, and should also be intrinsically faster than simplified SE-DFT algorithms.
This modeling approach is expected to be faster than any DFT (or simplified SE-DFT) algorithm.

## Running the code

The model is initialized by running 'model_creation.py'.  Look within this file to define the modeling parameters.  Note that this part of the code is written in non-accelerated Python, and may take a few minutes if mid-size molecules (~100 atoms) are included.

Monte-carlo optimization of the model parameters is performed by running 'data_creation.py', which is numba accelerated and closely resembles our expected performance on a personal computer CPU.  The mc_data object created in 'data_creation.py' can be used to examine the convergence and as a source of training data for AI-assisted optimization.

