# hop-field
Molecular dynamics with a tight binding field


The model is initialized by running 'model_creation.py'.  Look within this file to define the modeling parameters.  Note that this part of the code is written in non-accelerated Python, and may take a few minutes if mid-size molecules (~100 atoms) are included.

Monte-carlo optimization of the model parameters is performed by running 'data_creation.py', which is numba accelerated and closely resembles our expected performance on a personal computer CPU.  The mc_data object created in 'data_creation.py' can be used to examine the convergence and as a source of training data for AI-assisted optimization.

