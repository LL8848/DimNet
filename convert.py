# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:41:26 2022

@author: lnl5
"""

from itertools import product
import numpy as np
import copy
import pandas as pd

import matplotlib.pyplot as plt
round_n1, round_n2 = 4, 4  # 2, 4

import torch
from joblib import dump, load

def dimnet_to_correlation(model, verbosity=0):
    """
    Convert a trained DimNet to a piecewise function

    Parameters
    ----------
    model : Pipeline (defined in core.py)
        A pipeline that contains a trained DimNet and the x,y-scalers (typically saved with suffix .model)
    verbosity : int, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.

    """
    xscaler = model.xscaler
    yscaler = model.yscaler
    net = model.net
    
    # w1, w2, w3, b1, b2, b3: weight and bias of the DimNet
    with torch.no_grad():
        net.cpu()
        w1 = net.layer1.weight.numpy()
        b1 = net.layer1.bias.numpy()
        
        w2 = net.layer2.weight.numpy()
        b2 = net.layer2.bias.numpy()
        
        w3 = net.output.weight.numpy()
        b3 = net.output.bias.numpy()
    
    # u, s: vectors of the mean and the standard deviation of training samples
    u = xscaler.mean_   
    s = xscaler.scale_
    
    # y_factor: scaling factor for y (y_factor = y_max / eps_max)
    y_max = yscaler.scale_[0]
    y_factor = y_max
    
    
    if (verbosity > 2):
        print("-"*20)
        print("DimNet Parameters")
        print("-"*20)
        print("w1:")
        print(w1)
        print("b1:")
        print(b1)
        print("w2:")
        print(w2)
        print("b2:")
        print(b2) 
        print("w3:")
        print(w3)
        print("b3:")
        print(b3)
        print("-"*20)
        print("Scaling Parameters")
        print("-"*20)
        print("u:")
        print(u)
        print("s:")
        print(s)
        print("y_factor:")
        print(y_factor)
    
    
    pass # placeholder for the conversion calculation
    
    print("")
    print("-"*20)
    print("Algebraic Correlation")
    print("-"*20)
    
    """
    1D example
    
    z1 = log(Re**0.48) - 2.2840
    z2 = log(Re**-0.46) + 3.8497
    
    1) for z1 > 0, z2 > 0:
    -2.1e+03 + 6.2e+02 * (Re**-1.00) - 1.1e-21 * (Re**5.60)
    
    2) for z1 > 0, z2 <= 0:
    -2.1e+03 + 8.3e-01 * (Re**-0.45) - 5.0e-02 * (Re**-0.10)
    
    ...
    
    -----------------------------------------------------------------------
    2D example
    
    z1 = log(Re**0.48 * K**0.01) - 2.2840
    z2 = log(Re**-0.46 * K**0.02) + 3.8497
    
    1) for z1 > 0, z2 > 0:
    -2.1e+03 + 6.2e+02 * (Re**-1.00 * K**0.03) - 1.1e-21 * (Re**5.60 * K**0.03)
    
    2) for z1 > 0, z2 <= 0:
    -2.1e+03 + 8.3e-01 * (Re**-0.45 * K**0.03) - 5.0e-02 * (Re**-0.10 * K**0.03)
    
    ...
    """


def pw_predict():
    """
    used to validate the algebraic correlation printed by dimnet_to_pw
    copy the printed equations and paste here
    """



if __name__ == '__main__':
    path = 'test_pw/Clean_Smooth_200_[2, 1]_75.model'
    model = load(path)
    dimnet_to_correlation(model, verbosity=5)
    
    # compare preditions by the original DimNet and by the algebraic correlation
    
    # 1D example
    df1 = pd.read_csv('data_clean.csv')
    # y_dimnet = model.predict()
    
    
    # 2D example