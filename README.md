# DimNet:  Correlation Development Tool based on Neural Networks
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
* The code is written in Python 3, which depends on Python libraries including numpy, pandas, matplotlib, scipy, os, etc.  [Anaconda](https://www.anaconda.com/) is the recommended Python platform since it installs all dependencies.
* PyTorch, sci-kit learn

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



General Workflow
------
1.	Use the scripts in ```/lmpscript``` to run equilbration and NEMD simulations using LAMMPS on a HPC server.
2.  When computation completed, download the files from the server to:  ```./data/new```.
3.	Open a iPython-like terminal or a Juypter notebook, cd to ```./data/new```, import the modules in ```/src``` to post-process and analyze the data. Here are some tips:
    * Use ```vba = analyze()``` to quick-check all the viscosity output files in ```./data/new```. ```vba``` is a ```BatchData``` class that has a bunch of useful functions you can play with to analyze the viscosity data.
    * Use ```plot('filename')``` to visualize the output files that compute pressure, energy, etc. as a function of time. This is primarily to check if the equilibration or steady state is reached.
    * Use ```vd = vba.get(srate)``` to quickly retrieve a dataset for a state point where srate is the shear rate (1/s), e.g, 1e8, 1e9. ```vd``` is a ```ViscData``` class that also has a bunch of useful functions for analysis.
    * Deep steady-state check: use ```vd.acf()```, ```vd.ssplot()```, ```vd.setss1()```, etc
    * Check the [Cheatsheet](/cheatsheet.pdf) I made for the complete usage of the functions and classes.
4.	If steady-state is reached and desired statistical accuracy has been achieved, run ```copy('Path')``` to copy the visc_ file to the ```./data/visc```. Move all files in ```./data/new``` to ```./data/archive```. Otherwise go back to LAMMPS for longer simulation until obtaining the desired results.
5.	Create a Jupyter notebook to do analysis and write report using the modules in ```./src``` and the data in ```./data/visc```.  Export results if necessary for publication and making figures using other software.

Exemples 
------
* 1D problem: friction factor of flow in smooth tubes
* 2D problem: friction factor of flow in rough tubes


To-do 
------
- [ ] Complete the tutorials.


Contact 
------
Feel free to contact me should you have any questions:
Lingnan Lin, Ph.D.
Email: lingnan dot lin at nist dot gov
