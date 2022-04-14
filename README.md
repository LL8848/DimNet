# Correlation Development Tool based on Dimensionless Neural Networks (DimNet)
Lingnan Lin, April 2022

What this package does ...
------
* Construct a DimNet, and train it with your own dataset.
* Convert the trained DimNet to an **_explicit, algebraic, piecewise_** function.

What's DimNet?
------
* DimNet is a neural network that consists of an log-transformed input layer, two hidden layers (with activation functions of ReLU and Exp, respectively) and a linear output layer, as shown in the figure below.
* DimNet is equivalent to a piecewise function of power-law-like equations.

DimNet's structure
------
![Schematic of DimNet](/schematic.png)


Installation
------
Simply download the zip file and unzip to a folder where you can run Python.

License & Citation
------
This code is licensed under the Apache v2 license. Feel free to use all or portions for your research or related projects so long as you provide the following citation information:

*Lin, L., Gao, L., Kedzierski, M., Hwang, Y. 2022. A general model for flow boiling heat transfer in microfin tubes based on a new neural network architecture. Energy and AI. 8: 100151. https://doi.org/10.1016/j.egyai.2022.100151*

Dependencies
------
* The code is written in Python 3.  [Anaconda](https://www.anaconda.com/) is the recommended Python platform since it installs all basic dependencies (numpy, pandas, joblib, etc.)
* Additional: PyTorch, scikit-learn

Directory Structure
------
    .

    └── data                  # Data files
        ├── archive_nemd      # Input & output files of NEMD simulations in LAMMPS
        ├── archive_eq        # Input & output files for equilibration in LAMMPS
        ├── visc              # All the viscosity outputs
        ├── eq_system         # Equilibrated systems
        ├── new               # New files are placed here temporarily
    └── src                   # Codes for post-processing LAMMPS outputs
        ├── core.py           # Module for post-processing general output of LAMMPS ave/time fix
        ├── viscpost.py       # Module for post-processing viscosity data
        ├── rheologymodels.py # Module for various rheology models that are used to fit the shear viscosity
        ├── lmpcopy.py        # Module for organizing the files in different folders    
        ├── utility.py        # High-level functions for quick processing and analysis of results
    ├── examples              # Jupyter notebooks that call src modules to analyze the results



Usage
------
1.	Import the dataset and save it as a Pandas DataFrame ```df```.
2.  Use ```train_DimNet``` to construct and train a DimNet with ```df``` 
3.  

Exemples 
------
* 1D problem: friction factor of flow in smooth tubes
* 2D problem: friction factor of flow in rough tubes



Contact 
------
Feel free to contact me should you have any questions:
Lingnan Lin, Ph.D.
Email: lingnan dot lin at nist dot gov
