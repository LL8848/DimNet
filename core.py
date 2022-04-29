# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 17:44:37 2020

@author: Lingnan Lin, NIST, lingnan.lin@nist.gov
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from datetime import datetime
from collections import deque
import os

from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from joblib import dump



class DimNet(nn.Module):
    def __init__(self, n_in, n_units_1, n_units_2):
        super(DimNet, self).__init__()
        self.layer1 = nn.Linear(n_in, n_units_1)
        self.layer2 = nn.Linear(n_units_1, n_units_2)
        self.output = nn.Linear(n_units_2, 1)

    def forward(self, X):
        X = self.layer1(X)
        X = F.relu(X)
        X = self.layer2(X)
        X = torch.exp(X)
        X = self.output(X)
        return X

class Pipeline:
    def __init__(self, net, xscaler, yscaler):
        self.net = net
        self.xscaler = xscaler
        self.yscaler = yscaler

    def predict(self, X):
        """
        X: ndarray
        """
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        X = X.astype(np.float32)
        X = self.xscaler.transform(np.log(X))
        X = torch.tensor(X)

        self.net.cpu()
        self.net.eval()
        with torch.no_grad():
            y_pred = self.net(X).numpy()
        y_pred = self.yscaler.inverse_transform(y_pred)
        return y_pred


def plot_log(path,toShow=True,toSave=False, plot_ave=False):

    log = pd.read_csv(path, delim_whitespace=True)

    # plot the learning curve
    fig, ax = plt.subplots(nrows=2, ncols=1,sharex=False)
    ax[0].plot(log['EPOCH'],log['MAPE'].apply(lambda x: float(x[:-1])))
    # ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('MAPE [%]')
    ax[1].plot(log['EPOCH'],log['LOSS'])
    if plot_ave:
        ax[1].plot(log['EPOCH'],log['LOSS_AVE'],linestyle='--')
    ax[1].set_yscale('log')
    ax[1].set_yscale('log')
    # ax[1].set_xscale('log')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    plt.tight_layout()
    if toSave:
        path_new = path[:path.find('log')] + 'curve.png'
        plt.savefig(path_new,dpi=300)
    if toShow:
        plt.show()
    plt.close()
    plt.clf()



def compute_metrics(y_true, y_pred, verbose=True):

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # absolute percentage error
    perror = np.abs(y_true - y_pred) / y_true
    # mean absolute percentage error
    mape = np.mean(perror)
    # percentage predicted to within +-30%
    p30 = len(perror[perror <= 0.3]) / len(y_true)
     # percentage predicted to within +-50%
    p50 = len(perror[perror <= 0.5]) / len(y_true)

    if verbose:
        print(f"MAPE: {mape*100:.1f}%")
        print(f"P30: {p30*100:.1f}%")
        print(f"P50: {p50*100:.1f}%")
    return mape, p30, p50


def print_param(m):
    for name, param in m.named_parameters():
        if param.requires_grad:
            print(name, param.data)


def reset_params(m):

    if isinstance(m, torch.nn.Linear):
        m.reset_parameters()


def create_optimizer(SOLVER, net, lr=0.01):
    if SOLVER == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    elif SOLVER == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    elif SOLVER == 'LBFGS':
        optimizer = torch.optim.LBFGS(net.parameters(), lr=lr,
                                      max_iter=200,tolerance_grad=1e-9,
                                      tolerance_change=1e-12, history_size=200)
    return optimizer


def train_DimNet(
        df,col_x,col_y,NAME,N_HIDDEN,
        i_repeat = None,
        LR = 0.01,
        BATCH_SIZE = 128,
        MIN_EPOCH = 1,
        MAX_EPOCH = 100,
        INTERVAL_PRINT = 1,
        INTERVAL_SAVE = 10000,
        WIN_LEN = 10,
        CHANGE_RATE_TOLERANCE = 0.0001,
        DEVICE = 'cuda',
        TENSORBOARD = False,
        SOLVER = 'LBFGS',
        SHUFFLE = False,
        MANUAL_INIT = False,
        SAVE_INTER_MODEL = False,
        VERBOSITY = 0,
        SEED = None,
        weight0 = None,
        bias0 = None,

        ):


    ################## Preprocessing ##################

    path = NAME + '/'
    if type(i_repeat) == int:
        prefix = NAME + f"_{i_repeat}"
    else:
        prefix = NAME
    path_log = path+prefix+'_log.txt'

    DEVICE = torch.device(DEVICE)

    N_TRAIN = len(df)

    X = df[col_x].values.astype(np.float32)
    y = df[col_y].values.reshape(-1,1).astype(np.float32)

    X_scaler = StandardScaler()
    y_scaler = MaxAbsScaler()

    X_t = X_scaler.fit_transform(np.log(X))
    y_t = y_scaler.fit_transform(y)

    generator = torch.Generator(device='cpu')

    if SEED == None:
        generator.seed()
    else:
        generator.manual_seed(SEED)

    init_seed = generator.initial_seed()
    #%% Train
    #################### Training #####################

    if TENSORBOARD:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()

    N_IN = len(col_x)        # number of input units
    net = DimNet(N_IN, N_HIDDEN[0],N_HIDDEN[1]).to(DEVICE)

    # create directory if path doesn't exist
    if not os.path.exists(path):
        os.mkdir(path)
        # dump(X_scaler, os.path.join(path,'X_scaler'))
        # dump(y_scaler, os.path.join(path,'y_scaler'))



    # by default, read the parameters that was saved at the last epoch
    try:
        net.load_state_dict(torch.load(path+prefix+'.param'))
    except FileNotFoundError:
        print("No parameters loaded. Train from scratch.")
        torch.save(net.state_dict(), path+prefix+'_init.param')

    if MANUAL_INIT:
        print("Manually initialize the weights")
        with torch.no_grad():
            net.layer1.weight = torch.nn.Parameter(torch.tensor(weight0, device=DEVICE),
                                                   requires_grad=False)
            net.layer1.bias = torch.nn.Parameter(torch.tensor(bias0, device=DEVICE),
                                                 requires_grad=False)

    if MAX_EPOCH <= 0:
        print("Dry Run; Nothing is saved")
        model = Pipeline(net, X_scaler, y_scaler)
        return model


    optimizer = create_optimizer(SOLVER, net, lr=LR)

    loss_func = torch.nn.MSELoss()

    X_tensor = Variable(torch.tensor(X_t)).to(DEVICE)
    y_tensor = Variable(torch.tensor(y_t)).to(DEVICE)

    t0 = time()
    t_start = datetime.now()

    epoch_, loss_, mape_ = [], [], []

    window_change_rate = deque(maxlen=WIN_LEN)
    window_loss = deque(maxlen=WIN_LEN)
    change_rate = loss_ave_old = np.nan
    n_reset = 0
    normal = True
    num_non_improve = 0


    # read the log.txt to determine the last epoch #
    try:
        with open(path_log, 'r') as f_log:
            last_line = f_log.readlines()[-1]
            last_epoch = int(last_line.split()[0])
            mape_0 = float(last_line.split()[4][:-1])/100
            loss_old = float(last_line.split()[1])
            print(f"Resume from the last epoch {last_epoch}")
    except FileNotFoundError:
        last_epoch = 0
        loss_old = np.nan
        mape_0 = np.nan
        if VERBOSITY > 0:
            print("Created a new log.txt")
        with open(path_log, 'w') as f_log:
            f_log.write("     EPOCH        LOSS  LOSS_AVE    IMPROVE?   MAPE     TIME/s        ELAPSED       CHANGE\n")

    if VERBOSITY > 0:
        print("\n     EPOCH        LOSS   LOSS_AVE    IMPROVE?   MAPE     TIME/s        ELAPSED       CHANGE")

    print("Training in process ...... ")
    with open(path_log, 'a') as f_log:

        # train the net
        for i in range(last_epoch+1,last_epoch+MAX_EPOCH+1):

# =============================================================================
#           A Step of Training
# =============================================================================
            net.train()

            # Batch Gradient Descent
            # if SOLVER == 'BGD':
            #     optimizer.zero_grad()
            #     pred = net(X_tensor)
            #     loss = loss_func(pred, y_tensor)
            #     loss.backward()
            #     optimizer.step()

            # LBFGS
            if SOLVER == 'LBFGS':
                def closure():
                    optimizer.zero_grad()
                    prediction = net(X_tensor)
                    loss = loss_func(prediction, y_tensor)
                    loss.backward()
                    return loss
                optimizer.step(closure)

            if SOLVER in ['SGD','Adam']:
                if SHUFFLE:
                    permutation = torch.randperm(N_TRAIN,generator=generator)
                else:
                    permutation = torch.arange(N_TRAIN)
                for j in range(0, N_TRAIN, BATCH_SIZE):
                    optimizer.zero_grad()

                    indices = permutation[j:j+BATCH_SIZE]
                    batch_X, batch_y = X_tensor[indices], y_tensor[indices]

                    pred_j = net(batch_X)
                    loss = loss_func(pred_j, batch_y)
                    loss.backward()
                    optimizer.step()

# =============================================================================
#           Evaluation
# =============================================================================

            net.eval()
            prediction = net(X_tensor)
            loss = loss_func(prediction, y_tensor)

            loss_new = loss.data.to('cpu').numpy()
            # check if loss is lower than the previou N epoch
            improve = 'YES' if loss_new < loss_old else 'NO '
            # compute MAPE
            y_pred = prediction.data.to('cpu').numpy()
            y_pred = y_scaler.inverse_transform(y_pred)
            mape, _, _ = compute_metrics(y,y_pred,verbose=False)


            window_loss.append(loss_new)
            loss_ave = np.mean(window_loss)

            # because of the noise induced by SHUFFLE,
            # we can use moving average loss to calcualte change rate
            if SHUFFLE:
                change_rate = (loss_ave - loss_ave_old) / loss_ave_old
            else:
                change_rate = (loss_new - loss_old) / loss_old

            window_change_rate.append(change_rate)


# =============================================================================
#           Deal with Gradient Exploding
# =============================================================================

            if torch.isnan(loss):
                print("Overflowed (found NaN). Reset net parameters.")
                # re-initialize parameters
                net.apply(reset_params)
                # recreate an optimizer
                # because some optimizer, e.g. Adam and LBFGS, keep track of some stats)
                optimizer = create_optimizer(SOLVER, net, lr=LR)
                n_reset += 1
                if n_reset >= 30:
                    print(f"Reset params for {n_reset} times.  Program shut down")
                    normal = False
                    break

# =============================================================================
#           Print & Save
# =============================================================================
            # Always save net parameters in each step
            # torch.save(net.state_dict(), path+prefix+'.param')

            elapsed = time() - t0


            if (i % INTERVAL_PRINT == 0) & (i != 0):
                # sync with tensorbaord for real-time visualization
                if TENSORBOARD: writer.add_scalar("Loss/train", loss, i)
                # print info
                # print(f"{i:10d} {loss_new:15.10f}   {improve}    {mape*100:4.2f}%    {elapsed:5.2f}     {datetime.now()-t_start}   {change_rate*100:.4f}%")
                if VERBOSITY > 0:
                    print(f"{i:10d}  {loss_new:15.10f}  {loss_ave:.5e}   {improve}    {mape*100:4.2f}%    {elapsed:5.2f}     {datetime.now()-t_start}   {change_rate*100:.4f}%")
                f_log.write(f"{i:10d} {loss_new:15.10f}  {loss_ave:.5e}   {improve}    {mape*100:4.2f}%    {elapsed:5.2f}     {datetime.now()-t_start}   {change_rate*100:.4f}%\n")

            if INTERVAL_SAVE:
                if (i != 0) & (i % INTERVAL_SAVE == 0):
                    torch.save(net.state_dict(), path+prefix+'.param')

                    if SAVE_INTER_MODEL == True:
                        # torch.save(net, path+prefix+f'_{i}.dimnet')
                        model_i = Pipeline(net, X_scaler, y_scaler)
                        dump(model_i,path+prefix+f'_{i}.model')

# =============================================================================
#           Check Convergence
# =============================================================================
            if i > last_epoch+MIN_EPOCH:

                if SOLVER == 'LBFGS':

                    if change_rate >= -CHANGE_RATE_TOLERANCE:
                        num_non_improve += 1
                    else:
                        num_non_improve = 0

                else:
                    # if (np.any(np.array(window_change_rate) >= 0)) & (change_rate > - CHANGE_RATE_TOLERANCE):
                    # if np.mean(window_change_rate) > - CHANGE_RATE_TOLERANCE:
                    if change_rate >= -CHANGE_RATE_TOLERANCE:
                        num_non_improve += 1
                    else:
                        num_non_improve = 0

                if (num_non_improve > WIN_LEN):
                    print("Converged")
                    break


            # if (i >= last_epoch+MIN_EPOCH) & (len(window_change_rate) == WIN_LEN) & (np.mean(window_change_rate) > - CHANGE_RATE_TOLERANCE):
            #     print(f"converged (criterion: Loss doesn't change by more than {CHANGE_RATE_TOLERANCE*100:.4f}% in {WIN_LEN} consecutive epochs")
            #     break


# =============================================================================
#           Update Values
# =============================================================================
            t0 = time()
            loss_old = loss_new
            loss_ave_old = loss_ave
            epoch_.append(i)
            loss_.append(loss_new)
            mape_.append(mape)


    if TENSORBOARD: writer.flush()

    if normal:
        torch.save(net.state_dict(), path+prefix+'.param')
        torch.save(net, path+prefix+'.dimnet')
        print("Training completed!")
        print("To continue, run this cell again.\n\n")

        if VERBOSITY == 0:
            toShow = False
        else:
            toShow = True

        plot_log(path_log,toShow=toShow,toSave=True)

        model = Pipeline(net, X_scaler, y_scaler)
        dump(model, path+prefix+'.model')

    return model,mape,init_seed
