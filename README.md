# Correlation Development Tool based on DimNet
Developer: Lingnan Lin and Lei Gao.

What's DimNet?
------
* DimNet is a neural network that consists of an log-transformed input layer, two hidden layers (with activation functions of ReLU and Exp, respectively) and a linear output layer, as shown in the figure below.
* DimNet is equivalent to a piecewise function of power-law-like equations.

What this package does ...
------
* Construct a DimNet, and train it with your own dataset.
* Convert the trained DimNet to an **_explicit, algebraic, piecewise_** function.

DimNet's structure
------
![Schematic of DimNet](/schematic.png)


Installation & Usage
------
* Simply download the zip file and unzip to a folder where you can run Python.
* Follow the Tutorials, then 

License & Citation
------
This code is licensed under the Apache v2 license. Feel free to use all or portions for your research or related projects so long as you provide the following citation information:

*Lin, L., Gao, L., Kedzierski, M., Hwang, Y. 2022. A general model for flow boiling heat transfer in microfin tubes based on a new neural network architecture. Energy and AI. 8: 100151. https://doi.org/10.1016/j.egyai.2022.100151*

Dependencies
------
* The code is written in Python 3.  [Anaconda](https://www.anaconda.com/) is the recommended Python platform since it installs all basic dependencies (numpy, pandas, joblib, etc.)
* Additional: PyTorch, scikit-learn
* The tutorials are run in Jupyter Notebook, which is included in Anaconda.



Tutorials 
------
* [Example_1D](/Example_1D.ipynb): friction factor of flow in smooth tubes
* [Example_2D](/Example_2D.ipynb): friction factor of flow in rough tubes



Contact 
------
Feel free to contact me should you have any questions:
Lingnan Lin, Ph.D.
Email: lingnan dot lin at nist dot gov
