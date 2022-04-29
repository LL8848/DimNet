# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:41:26 2022

@author: lnl5
"""

from itertools import product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from joblib import load

Subscripts = {1:'₁',
              2:'₂',
              3:'₃',
              4:'₄',
              5:'₅',
              6:'₆',
              7:'₇',
              8:'₈',
              9:'₉',
              0:'₀'}

def pi_str(num):
    """
    Generate a string for Π with a subscript of num
    """
    split_num = [int(a) for a in str(num)]
    
    sub = []
    for x in split_num:
        sub.append(Subscripts[x])
    
    s = ''.join(sub)
    return('Π' + s)



def dimnet_to_correlation(model, col_x=None, col_y=None, digits_exp=2, digits=3, verbose=10):  
    """
    Convert a trained DimNet to a piecewise function

    Parameters
    ----------
    model : Pipeline (defined in core.py)
        A pipeline that contains a trained DimNet and the x,y-scalers (typically saved with suffix .model).
    col_x : list of string
        Names of the input variables. (default: ['x[0]','x[1]',...])
    col_y : list of string
        Names of the output variables. (default: ['y'])
    digits_exp : int
        number of digits for exponents. (default: 2)
    digits : int
        number of digits for non-exponent coefficients. (default: 3)
    verbose : int, optional
        controls the verbosity. The higher, the more messages printed out. (default: 0)

    Returns
    -------
    None.

    """

    # u, s: vectors of the mean and the standard deviation of training samples
    u = model.xscaler.mean_
    s = model.xscaler.scale_
    # y_factor: scaling factor for y (y_factor = y_max / eps_max)
    y_max = model.yscaler.scale_[0]
    y_factor = y_max

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
    
    L, M, N = len(w1[0]), len(w2[0]), len(w3[0])  # corresponding to indexes of i, j, k
    print("-" * 80)
    print(f"DimNet Config: {L} - {M} - {N} - 1")
    
    if col_x == None:
        # col_x = [pi_str(l+1) for l in range(L)]
        col_x = [f'x[{l}]' for l in range(L)]
        
    if col_y == None:
        # col_y = ['Π₀']
        col_y = ['y']

    print("-" * 80)
    print("Input variable(s): " + " ".join(col_x))
    print("Output variable(s): " + " ".join(col_y))


    if verbose > 2:
        print("-" * 80)
        print("DimNet Parameters")
        print("-" * 80)
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
        print("-" * 80)
        print("Scaling Parameters")
        print("-" * 80)
        print("u:")
        print(u)
        print("s:")
        print(s)
        print("y_factor:")
        print(y_factor)

    # Analysis for each possible hyperplane: for hyperplane divided by ReLU function with subsection of 2 ** M
    hyperplane = [[1, 0] for _ in range(M)]
    delta = []  # delta is a binary variable defined delta=1 for z>0, delta=0 for z<=0
    for d in product(*hyperplane):
        delta.append(list(d))
    delta_sign = [[0 for _ in range(M)] for _ in range(2 ** M)]
    for d in range(len(delta)):
        for j in range(M):
            if delta[d][j] == 1:
                delta_sign[d][j] = '>'
            else:
                delta_sign[d][j] = '<='

    print("-" * 80)
    print("Algebraic Correlation:\n")
    # hyperplanes
    for j in range(M):

        zb = b1[j] - sum(w1[j] * (u / s))

        print(f"z{j+1:d} = {zb:.{digits}e} + ", end="")
        for i in range(L):
            if i == 0:
                if L == 1:
                    print(f'ln({col_x[i]}**{w1[j][i] / s[i]:.{digits_exp}f})')
                else:
                    print(f'ln({col_x[i]}**{w1[j][i] / s[i]:.{digits_exp}f}', end="")

            elif i < L-1:
                print(f' * {col_x[i]}**{w1[j][i] / s[i]:.{digits_exp}f}', end="")
            else:
                print(f' * {col_x[i]}**{w1[j][i] / s[i]:.{digits_exp}f})')

    # sub-equations
    for d in range(len(delta)):
        print(" ")
        delta_diag = np.diag(delta[d])
        P = w3[0] * np.exp(((w2 @ delta_diag) @ (b1 - (w1 @ (u / s))) + b2)) * y_factor
        Q = ((w2 @ delta_diag) @ w1) / s
        for j in range(M):
            if j == 0:
                print(f"if z{j + 1:d} {delta_sign[d][j]:s} 0", end="")
            elif j < M-1:
                print(f" and z{j + 1:d} {delta_sign[d][j]:s} 0", end="")
            else:
                print(f" and z{j + 1:d} {delta_sign[d][j]:s} 0:")
        p0 = b3[0]*y_factor
        print(f"    {col_y[0]} = {p0:.{digits}e}", end="")        
        for k in range(N):
            print(f" + {P[k]:.{digits}e} * (", end="")
            for i in range(L):
                if i != L - 1:
                    print(f"{col_x[i]}**{Q[k][i]:.{digits_exp}f} * ", end="")
                else:
                    print(f"{col_x[i]}**{Q[k][i]:.{digits_exp}f})", end="")
    print("")
    print("-" * 80)

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


def pw_predict(x):
    """
    used to validate the algebraic correlation printed by dimnet_to_pw
    copy the printed equations and paste here
    """
    from math import log as ln

    z1 = -2.284e+00 + ln(x[0]**0.48)
    z2 = 3.850e+00 + ln(x[0]**-0.49)
     
    if z1 > 0 and z2 > 0:
        y = -2.083e-03 + 6.171e+01 * (x[0]**-0.99) + 1.093e-21 * (x[0]**5.60) 
    if z1 > 0 and z2 <= 0:
        y = -2.083e-03 + 8.262e-01 * (x[0]**-0.45) + 4.954e-02 * (x[0]**-0.10) 
    if z1 <= 0 and z2 > 0:
        y = -2.083e-03 + 7.485e+00 * (x[0]**-0.54) + 6.748e-22 * (x[0]**5.71) 
    if z1 <= 0 and z2 <= 0:
        y = -2.083e-03 + 1.002e-01 * (x[0]**0.00) + 3.060e-02 * (x[0]**0.00)   
    
    return y


if __name__ == '__main__':
    # 1D examples
    path = 'data/Evo_Noisy_Smooth_250_[2, 2]_50000.model'
    # path = 'data/Noisy_Smooth_200_[2, 2]_75.model'
    # path = 'data/Noisy_Smooth_200_[2, 3]_77.model'
    # path = 'data/Noisy_Smooth_200_[3, 1]_17.model'
    # path = 'data/Noisy_Smooth_200_[3, 2]_98.model'
    # path = 'data/Noisy_Smooth_200_[3, 3]_27.model'
    
    # 2D examples
    # path = 'data/Noisy_Rough_1200_[2, 2]_5.model'
    # path = 'data/Noisy_Rough_1200_[3, 2]_34.model'
    # path = 'data/Noisy_Rough_1200_[3, 3]_8.model'
    
    df_train = pd.read_csv("data/data_smooth.csv")    
    # df_train = pd.read_csv("data/data_rough.csv")
    
    col_x = ['Re'] # list of the names of the input variables
    col_y = ['f'] # list of the names of the output variables
    model = load(path)
    
    dimnet_to_correlation(model,
                          digits_exp=2, digits=3, verbose=0)

    # validation:
    
    print("Original DimNet vs. Converted Piecewise Function:\n")
    
    y_nn = model.predict(df_train[col_x].values).flatten()
    y_label = df_train.f.values
    
    y_pw = []

    for index, row in df_train.iterrows():
        y_pw.append(pw_predict(row[col_x].values))
    y_pw = np.array(y_pw)
    
    diff = (y_nn - y_pw)/y_nn

    print(f"mean abs. difference = {np.mean(np.abs(diff)):.2e}")
    print(f"max difference  = {np.max(diff):.2e}")
    print(f"min difference  = {np.min(diff):.2e}")    
    plt.hist(diff)
    plt.show()
    
    plt.scatter(y_nn,diff)
    plt.xlabel('y_nn')
    plt.ylabel('(y_nn - y_pw)/y_nn')

    from core import compute_metrics
    mape_pw,_,_ = compute_metrics(y_label, y_pw, verbose=False)
    print(f"mape_pw = {mape_pw*100:.2f}%")
    mape_nn,_,_ = compute_metrics(y_label, y_nn, verbose=False)
    print(f"mape_nn = {mape_nn*100:.2f}%")        

